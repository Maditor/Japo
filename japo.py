"""
Japo – real-time subtitles for Japanese audio.
Captures system audio (WASAPI loopback), transcribes it with faster-whisper,
translates it with an LLM (Cloudflare / Gemini) or machine translation,
and shows the result in a sidebar window.

Author: Maditor - https://github.com/Maditor/Japo
"""
import os
import queue
import threading
import time
from datetime import datetime

import numpy as np

# ============================ SETTINGS ============================
MODEL_SIZE = "large-v3-turbo"  # use "small" on slower machines
DEVICE = "cuda"                # "cuda" (NVIDIA GPU) or "cpu"
COMPUTE_TYPE = "int8_float16"  # fits 4 GB VRAM
SOURCE_LANG = "ja"
TARGET_LANG = "vi"             # "en" for English subtitles
SILENCE_THRESHOLD = 0.008      # only used when AUTO_GAIN = False
AUTO_GAIN = True               # works at low playback volume
SENSITIVITY = 0.22             # lower = catch quiet speech, higher = ignore background music
SILENCE_SEC = 0.6              # pause length that ends a line
MAX_SEGMENT_SEC = 8.0          # force a cut after this many seconds
MIN_SEGMENT_SEC = 0.8          # ignore shorter segments
SHOW_EN = True                 # also show an English machine translation
CLOUDFLARE_MODEL = "@cf/google/gemma-4-26b-a4b-it"  # used when cloudflare.txt exists
GEMINI_MODEL = "gemini-2.5-flash"                   # used when gemini_key.txt exists
FILTER_INTERJECTIONS = True    # hide lines that are only short vocal sounds
STRICT_THANKS = True           # drop low-confidence "arigatou..." hallucinations
BEAM_SIZE = 1                  # 1 = fastest, 3-5 = slightly more accurate
MERGE_MAX_SEC = 15.0           # merge queued segments when falling behind
CONTEXT_LINES = 6              # previous lines sent to the LLM as context
CONTEXT_RESET_SEC = 20         # forget context after a gap this long (seeking / pausing)
MALE_MAX_HZ = 160              # median voice pitch below this -> male
FEMALE_MIN_HZ = 175            # above this -> female (in between = unknown)
# ================================================================

TARGET_RATE = 16000
import sys
FROZEN = getattr(sys, "frozen", False)   # running as a PyInstaller build
# Folder of the exe / script: cloudflare.txt, gemini_key.txt and model/ live here
APP_DIR = os.path.dirname(sys.executable) if FROZEN else os.path.dirname(os.path.abspath(__file__))

# Windowed build has no console: write logs to a file
if sys.stdout is None or sys.stderr is None:
    try:
        _log = open(os.path.join(APP_DIR, "japo_log.txt"), "w", encoding="utf-8", buffering=1)
    except OSError:
        _log = open(os.devnull, "w")
    sys.stdout = sys.stdout or _log
    sys.stderr = sys.stderr or _log

# Use a manually downloaded model/ folder if present
if os.path.isfile(os.path.join(APP_DIR, "model", "model.bin")):
    MODEL_SIZE = os.path.join(APP_DIR, "model")


import re as _re

# Phrases Whisper tends to hallucinate from non-speech audio
SUSPICIOUS = {
    "ありがとうございました", "ありがとうございます", "ありがとう", "ご視聴ありがとうございました",
    "おやすみなさい", "お疲れ様でした", "お疲れ様です", "ごちそうさまでした",
    "チャンネル登録お願いします", "チャンネル登録よろしくお願いします", "字幕", "では",
    "また会いましょう", "またね", "バイバイ", "さようなら",
}
_PUNCT = "。、．，.,!！?？…‥・ー〜～ 　「」『』()（）\"'"
_INTERJECTION_RE = _re.compile(r"^[あぁいぃうぅえぇおぉんンっッはハぁァアイウエオふフひヒ"
                       r"ー〜～…‥・。、!！?？♡♥\s]+$")


# Short words with real meaning (never filtered)
REAL_SHORT = {"はい", "うん", "ううん", "いい", "いや", "ええ", "おい", "いえ", "あい", "いいえ",
              "はいはい", "うんうん", "いいい", "ええっ", "えっ", "へえ", "ほう", "おお"}


def _norm(t):
    return "".join(c for c in t if c not in _PUNCT)


# Short expressive words that are often repeated on purpose (never drop these for repetition)
REPEATABLE_WORDS = ["イク", "いく", "イッちゃう", "いっちゃう", "イっちゃう", "イキそう", "いきそう",
                    "気持ちいい", "きもちいい", "ダメ", "だめ", "やめて", "もっと", "すごい",
                    "出る", "出ちゃう", "待って", "やだ", "いや", "好き"]


def collapse_repeats(text):
    """'イクイクイクイク' -> 'イク、イク、イク…'. Returns None if the text is not such a repeat."""
    n = _norm(text)
    for w in sorted(REPEATABLE_WORDS, key=len, reverse=True):
        if len(n) >= 2 * len(w) and n == w * (len(n) // len(w)):
            k = len(n) // len(w)
            return "、".join([w] * min(k, 3)) + ("…" if k > 3 else "")
    return None


def keep_segment(text, avg_logprob, no_speech_prob, compression_ratio):
    """Decide whether to keep a transcribed segment."""
    t = (text or "").strip()
    if not t:
        return False
    n = _norm(t)
    if not n:
        return False
    # 1) interjection-only lines
    if FILTER_INTERJECTIONS and _INTERJECTION_RE.match(t) and len(n) <= 12 and n not in REAL_SHORT:
        return False
    # 2) common hallucinations: keep only when highly confident
    if STRICT_THANKS:
        base = n
        repeated = any(base == p * k for p in SUSPICIOUS for k in (2, 3, 4))
        if base in SUSPICIOUS or repeated:
            if repeated or avg_logprob < -0.3 or no_speech_prob > 0.15:
                return False
    # 3) repetitive or low-confidence output (expressive repeats are allowed)
    if compression_ratio > 2.4 and not collapse_repeats(t):
        return False
    if no_speech_prob > 0.6 and avg_logprob < -0.8:
        return False
    if avg_logprob < -1.2:
        return False
    return True


_ROMAJI = {"fn": None, "tried": False, "error": None}
_ROMAJI_FIX = {"konnichiha": "konnichiwa", "konbanha": "konbanwa",
               "oanichan": "oniichan", "oanechan": "oneechan", "oanisan": "oniisan",
               "oanesan": "oneesan", "watakushi": "watashi"}


def to_romaji(text):
    """Japanese -> romaji. Returns "" on failure."""
    if not _ROMAJI["tried"]:
        _ROMAJI["tried"] = True
        try:
            import cutlet
            k = cutlet.Cutlet()
            k.use_foreign_spelling = False
            _ROMAJI["fn"] = k.romaji
        except Exception as e:
            print("cutlet unavailable, trying pykakasi:", e, flush=True)
            try:
                import pykakasi
                kk = pykakasi.kakasi()
                _ROMAJI["fn"] = lambda t: " ".join(
                    x["hepburn"] for x in kk.convert(t) if x["hepburn"].strip())
            except Exception as e2:
                print("No romaji converter available:", e2, flush=True)
                _ROMAJI["error"] = "Romaji unavailable – run setup.bat to install cutlet"
    fn = _ROMAJI["fn"]
    if not fn:
        return ""
    try:
        out = fn(text)
        for a, b in _ROMAJI_FIX.items():
            out = out.replace(a, b).replace(a.capitalize(), b.capitalize())
        return out
    except Exception:
        return ""


def estimate_pitch(audio, sr=16000):
    """Median voice pitch (Hz) of a segment via frame autocorrelation, or None."""
    frame, hop = int(0.04 * sr), int(0.02 * sr)
    if audio is None or len(audio) < frame * 3:
        return None
    n = 1 + (len(audio) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    fr = audio[idx] * np.hanning(frame)[None, :]
    energy = (fr ** 2).mean(axis=1)
    fr = fr[energy > max(np.percentile(energy, 40), 1e-7)]
    if len(fr) < 5:
        return None
    fr = fr - fr.mean(axis=1, keepdims=True)
    spec = np.fft.rfft(fr, 2 * frame, axis=1)
    ac = np.fft.irfft(np.abs(spec) ** 2, axis=1)[:, :frame]
    ac = ac / np.maximum(ac[:, :1], 1e-12)
    lo, hi = int(sr / 400), int(sr / 70)          # 70-400 Hz
    seg = ac[:, lo:hi]
    lag = seg.argmax(axis=1) + lo
    voiced = seg.max(axis=1) > 0.45
    if voiced.sum() < 5:
        return None
    return float(np.median(sr / lag[voiced]))


def classify_voice(audio):
    """'f', 'm' or None."""
    try:
        f0 = estimate_pitch(audio)
    except Exception:
        return None
    if f0 is None:
        return None
    if f0 >= FEMALE_MIN_HZ:
        return "f"
    if f0 <= MALE_MAX_HZ:
        return "m"
    return None


def read_cloudflare():
    """cloudflare.txt: line 1 = Account ID, line 2 = API token
    (or ACCOUNT_ID=... / API_TOKEN=...)."""
    p = os.path.join(APP_DIR, "cloudflare.txt")
    try:
        with open(p, encoding="utf-8-sig") as f:
            lines = [l.strip() for l in f if l.strip()]
    except OSError:
        return None
    vals = {}
    plain = []
    for l in lines:
        if "=" in l:
            k, v = l.split("=", 1)
            vals[k.strip().upper()] = v.strip().strip('"')
        else:
            plain.append(l)
    acc = vals.get("ACCOUNT_ID") or (plain[0] if len(plain) > 0 else None)
    tok = vals.get("API_TOKEN") or (plain[1] if len(plain) > 1 else None)
    if acc and tok:
        return acc, tok
    print("cloudflare.txt is missing the Account ID or API token", flush=True)
    return None


def read_gemini_key():
    p = os.path.join(APP_DIR, "gemini_key.txt")
    try:
        with open(p, encoding="utf-8") as f:
            key = f.read().strip()
        return key or None
    except OSError:
        return None


def add_cuda_dll_paths():
    """Let Windows find the pip-installed cuBLAS/cuDNN DLLs."""
    if os.name != "nt":
        return
    import site
    bases = []
    if getattr(sys, "_MEIPASS", None):       # frozen build: DLLs are in _internal
        bases.append(sys._MEIPASS)
    try:
        bases += list(site.getsitepackages())
        bases.append(site.getusersitepackages())
    except Exception:
        pass
    for base in bases:
        for sub in ("nvidia/cublas/bin", "nvidia/cudnn/bin", "nvidia/cuda_runtime/bin"):
            p = os.path.join(base, sub)
            if os.path.isdir(p):
                try:
                    os.add_dll_directory(p)
                except Exception:
                    pass
                os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")


# ---------------------------------------------------------------- audio
class LoopbackCapture:
    """Capture audio playing on the default output device (WASAPI loopback)."""

    def __init__(self, out_q):
        import pyaudiowpatch as pyaudio
        self.pyaudio = pyaudio
        self.out_q = out_q
        self.p = pyaudio.PyAudio()
        self.device = self._find_loopback()
        self.rate = int(self.device["defaultSampleRate"])
        self.channels = int(self.device["maxInputChannels"])
        self.stream = None

    def _find_loopback(self):
        pa = self.pyaudio
        wasapi = self.p.get_host_api_info_by_type(pa.paWASAPI)
        speakers = self.p.get_device_info_by_index(wasapi["defaultOutputDevice"])
        if speakers.get("isLoopbackDevice"):
            return speakers
        for lb in self.p.get_loopback_device_info_generator():
            if speakers["name"] in lb["name"]:
                return lb
        raise RuntimeError("No loopback device found for the default speakers.")

    def _callback(self, in_data, frame_count, time_info, status):
        self.out_q.put(in_data)
        return (None, self.pyaudio.paContinue)

    def start(self):
        self.stream = self.p.open(
            format=self.pyaudio.paInt16,
            channels=self.channels,
            rate=self.rate,
            input=True,
            input_device_index=self.device["index"],
            frames_per_buffer=int(self.rate * 0.1),
            stream_callback=self._callback,
        )
        self.stream.start_stream()

    def stop(self):
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            self.p.terminate()
        except Exception:
            pass


def to_mono_16k(raw, rate, channels):
    from scipy.signal import resample_poly
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        x = x.reshape(-1, channels).mean(axis=1)
    if rate != TARGET_RATE:
        g = np.gcd(rate, TARGET_RATE)
        x = resample_poly(x, TARGET_RATE // g, rate // g).astype(np.float32)
    return x


# ------------------------------------------------------------ engine
class SubtitleEngine:
    def __init__(self, ui_q):
        self.ui_q = ui_q
        self.audio_q = queue.Queue()
        self.seg_q = queue.Queue()
        self.tr_q = queue.Queue()     # transcribed lines waiting for translation
        self._cache, self._cooldown, self._context = {}, {}, []
        self.running = True
        self.paused = False
        self.model = None
        self.device = DEVICE
        self.capture = None
        self.gemini_key = None
        self.cloudflare = None
        self.gemini_note = None   # translation warning shown in the status area
        self.uncensored = False
        self.detect_gender = True
        self.movie_context = ""      # title / plot / characters typed by the viewer
        self.failed = False
        self.level = 0.0       # current input level (0..1) for the meter
        self.listening = False
        self.busy = False      # transcription in progress

    def status(self, text):
        print(text, flush=True)
        self.ui_q.put(("status", text))

    # --- model
    def load_model(self):
        from faster_whisper import WhisperModel
        if self.device == "cuda":
            try:
                add_cuda_dll_paths()
                self.status("Loading model on GPU… (first run downloads it)")
                self.model = WhisperModel(MODEL_SIZE, device="cuda", compute_type=COMPUTE_TYPE)
                return
            except Exception as e:
                self.ui_q.put(("error", f"GPU unavailable ({e}). Using CPU."))
                self.device = "cpu"
        self.status("Loading model on CPU…")
        self.model = WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=6)

    def fallback_to_cpu(self, err):
        self.ui_q.put(("error", f"GPU error: {err}. Switched to CPU."))
        self.device = "cpu"
        self.model = None
        self.load_model()

    # --- threads
    def start(self):
        threading.Thread(target=self._init_and_run, daemon=True).start()

    def _init_and_run(self):
        try:
            self.gemini_key = read_gemini_key()
            self.cloudflare = read_cloudflare()
            self.load_model()
            self.capture = LoopbackCapture(self.audio_q)
            self.capture.start()
        except Exception as e:
            self.failed = True
            self.status("Startup failed")
            self.ui_q.put(("error", f"Startup failed: {e}"))
            return
        threading.Thread(target=self._segmenter, daemon=True).start()
        threading.Thread(target=self._transcriber, daemon=True).start()
        threading.Thread(target=self._translator_loop, daemon=True).start()
        self.listening = True
        tr = ("Cloudflare Gemma" if self.cloudflare
              else "Gemini" if self.gemini_key else "machine translation")
        self.status(f"Listening: {self.capture.device['name']} | {self.device.upper()} | {tr}")

    def _segmenter(self):
        """Split audio into lines at pauses. The silence threshold adapts to the
        playback level, so quiet playback still works."""
        from collections import deque
        rate, ch = self.capture.rate, self.capture.channels
        buf, speech, silence = [], False, 0.0
        recent = deque(maxlen=150)   # levels of the last ~15 s
        while self.running:
            try:
                raw = self.audio_q.get(timeout=0.3)
                chunk = to_mono_16k(raw, rate, ch)
                dur = len(chunk) / TARGET_RATE
            except queue.Empty:
                chunk, dur = None, 0.3  # no audio (playback paused)

            if self.paused:
                buf, speech, silence = [], False, 0.0
                continue

            if chunk is None:
                self.level = 0.0
                if speech:
                    silence += dur
            else:
                rms = float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) else 0.0
                if AUTO_GAIN:
                    if rms > 1e-5:                    # skip digital silence
                        recent.append(rms)
                    if len(recent) >= 10:
                        arr = np.array(recent)
                        loud = float(np.percentile(arr, 90))
                        noise = float(np.percentile(arr, 15))
                        thr = max(0.0003, noise * 2.0, loud * SENSITIVITY)
                    else:
                        loud, thr = 0.02, 0.001
                    self.level = min(1.0, rms / max(loud, 1e-6))
                else:
                    thr = SILENCE_THRESHOLD
                    self.level = min(1.0, rms / (SILENCE_THRESHOLD * 4))
                if rms > thr:
                    speech, silence = True, 0.0
                else:
                    silence += dur
                if speech:
                    buf.append(chunk)

            total = sum(len(b) for b in buf) / TARGET_RATE
            if speech and (silence >= SILENCE_SEC or total >= MAX_SEGMENT_SEC):
                if total >= MIN_SEGMENT_SEC:
                    audio = np.concatenate(buf)
                    if AUTO_GAIN:                     # normalize level before transcription
                        peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
                        if peak > 1e-5:
                            audio = (audio * min(60.0, 0.9 / peak)).astype(np.float32)
                    self.seg_q.put((datetime.now(), audio))
                buf, speech, silence = [], False, 0.0

    def _transcribe(self, audio):
        segments, _ = self.model.transcribe(
            audio,
            language=SOURCE_LANG,
            beam_size=BEAM_SIZE,
            best_of=1,
            temperature=0.0,
            vad_filter=True,
            vad_parameters=dict(threshold=0.55, min_speech_duration_ms=300,
                                min_silence_duration_ms=400, speech_pad_ms=200),
            condition_on_previous_text=False,
            without_timestamps=True,
            # Prime Whisper to write vocal sounds as interjections instead of inventing sentences
            initial_prompt="あっ…んっ、イクっ…はぁ…はぁ…。うん。",
            compression_ratio_threshold=2.2,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.5,
        )
        parts = []
        for s in segments:
            t = s.text.strip()
            if keep_segment(t, s.avg_logprob, s.no_speech_prob, s.compression_ratio):
                t = collapse_repeats(t) or t
                parts.append(t)
        return "".join(parts).strip()

    def _transcriber(self):
        while self.running:
            try:
                ts, audio = self.seg_q.get(timeout=0.5)
            except queue.Empty:
                continue
            # Falling behind: transcribe queued segments in one pass
            pieces = [audio]
            total = len(audio) / TARGET_RATE
            while total < MERGE_MAX_SEC:
                try:
                    _, nxt = self.seg_q.get_nowait()
                except queue.Empty:
                    break
                pieces.append(np.zeros(int(0.3 * TARGET_RATE), dtype=np.float32))
                pieces.append(nxt)
                total += len(nxt) / TARGET_RATE + 0.3
            if len(pieces) > 1:
                audio = np.concatenate(pieces)
            self.busy = True
            try:
                text = self._transcribe(audio)
            except Exception as e:
                if self.device == "cuda":
                    self.fallback_to_cpu(e)
                    try:
                        text = self._transcribe(audio)
                    except Exception as e2:
                        self.ui_q.put(("error", str(e2)))
                        continue
                else:
                    self.ui_q.put(("error", str(e)))
                    continue
            finally:
                self.busy = False
            if not text or all(c in "。、！？…・ 　!?." for c in text):
                continue
            gender = classify_voice(audio) if self.detect_gender else None
            # Drop the same very short line repeated within 30 s (e.g. a stream of "うん")
            short = _norm(text)
            last = getattr(self, "_last_short", (None, 0))
            if len(short) <= 3 and short == last[0] and time.time() - last[1] < 30:
                continue
            self._last_short = (short, time.time())
            self.tr_q.put((ts, text, gender))

    def _translator_loop(self):
        """Translate on a separate thread so transcription never waits."""
        while self.running:
            try:
                ts, text, gender = self.tr_q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                vi, en = self.translate_line(text, gender)
            except Exception as e:
                vi, en = f"(translation error: {e})", None
            self.ui_q.put(("line", ts, text, vi, en, to_romaji(text), gender))

    # ------------------------------------------------------------ translation
    def translate_line(self, text, gender=None):
        """Return (main translation, English translation or None)."""
        if not hasattr(self, "_context"):
            self._context = []
        now = time.time()
        if now - getattr(self, "_last_line_at", now) > CONTEXT_RESET_SEC:
            self._context = []          # new scene: don't carry over old context
        self._last_line_at = now
        en_future = None
        if SHOW_EN and TARGET_LANG != "en":
            from concurrent.futures import ThreadPoolExecutor
            if not hasattr(self, "_pool"):
                self._pool = ThreadPoolExecutor(max_workers=2)
            en_future = self._pool.submit(self.machine_translate, text, "en")
        vi = None
        ais = []
        if self.cloudflare:
            ais.append(("Cloudflare", self._translate_cloudflare))
        if self.gemini_key:
            ais.append(("Gemini", self._translate_gemini))
        notes = []
        for name, fn in ais:
            try:
                vi = fn(text, gender)
                break
            except Exception as e:
                code = getattr(getattr(e, "response", None), "status_code", None)
                detail = ""
                try:
                    detail = e.response.text[:300]
                except Exception:
                    pass
                print(f"{name} ERROR:", code or "", e, detail, flush=True)
                notes.append(f"{name} error ({code or type(e).__name__})")
        self.gemini_note = (", ".join(notes) + ("; using machine translation" if not vi else "")) if notes else None
        if not vi:
            vi = self.machine_translate(text, TARGET_LANG)
        en = None
        if en_future is not None:
            try:
                en = en_future.result(timeout=10)
            except Exception:
                en = None
        self._context.append((text, vi, gender))
        self._context = self._context[-CONTEXT_LINES:]
        return vi, en

    def _llm_prompt(self, text, gender=None):
        T = {"vi": "Vietnamese", "en": "English"}.get(TARGET_LANG, TARGET_LANG)
        who = {"f": "[female voice] ", "m": "[male voice] "}
        prev = "\n".join(f"- {who.get(g, '')}{ja}  =>  {tr}" for ja, tr, g in self._context)
        speaker = {"f": "female", "m": "male"}.get(gender)
        speaker_line = (f"\nSpeaker: {speaker} voice (guessed from pitch, usually right)." if speaker else "")
        system = (
            f"You are a professional subtitle translator. Translate Japanese movie dialogue "
            f"into natural, spoken {T}."
        )
        about = self.movie_context.strip()[:1500]
        about_block = (f"About this film (from the viewer; use it for names, relationships, "
                       f"tone and forms of address):\n{about}\n" if about else "")
        prompt = f"""{system}
{about_block}{f"Previous lines, for context only (do NOT translate them again):{chr(10)}{prev}{chr(10)}" if prev else ""}
Line to translate:
{text}{speaker_line}

Rules:
1. Write natural, casual spoken {T} like professional movie subtitles, not word-for-word.
2. Choose {T} pronouns and forms of address that fit the speakers' relationship, and keep them consistent with the previous lines.
3. The Japanese comes from speech recognition and may contain misheard words: guess the most plausible meaning from context.
4. Vocal sounds, sound effects and onomatopoeia: render them briefly as an equivalent expression in {T}.
   Use the speaker's gender (when given) to choose natural {T} pronouns and self-reference, and to tell who is talking to whom.
5. Japanese often uses kinship words as forms of address: a host or stranger may call a woman "お母さん/お母様" (the mother),
   a young man "お兄さん", an older woman "お姉さん". Translate these as the addressee's role, not as the speaker's own family,
   unless the context says they really are related. A single line may contain two speakers (question + answer).
6. Keep the original tone, intensity and emotion. Keep punctuation style (…, ?!, !!).
7. Output ONLY the {T} translation as one line. No notes, no quotes, no original text.{self._uncensored_rules(T)}"""
        return system, prompt

    def _uncensored_rules(self, T):
        if not self.uncensored:
            return ""
        return f"""

Mode: uncensored. The dialogue may contain mature language.
- Translate faithfully; never soften, censor, euphemize or omit words.
- Use the natural colloquial {T} a native speaker would actually say, slang included.
- Keep the speaker's tone and intensity unchanged.
- Do not add meaning that is not in the original."""

    @staticmethod
    def _clean_llm(out):
        import re
        out = re.sub(r"<think>.*?</think>", "", out or "", flags=re.S).strip()
        lines = [l.strip() for l in out.splitlines() if l.strip()]
        out = lines[-1] if lines else ""
        return out.strip().strip('"「」“”')

    @staticmethod
    def _extract_text(data):
        """Extract the reply text from the various Cloudflare response shapes."""
        res = data.get("result", data) if isinstance(data, dict) else data
        if res is None:
            res = data
        def as_text(c):
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                return "".join(as_text(x) for x in c)
            if isinstance(c, dict):
                return c.get("text") or c.get("content") or c.get("response") or ""
            return ""
        if isinstance(res, dict):
            ch = res.get("choices")
            if ch:
                msg = ch[0].get("message") or ch[0].get("delta") or {}
                return as_text(msg.get("content")) or as_text(ch[0].get("text"))
            return as_text(res.get("response")) or as_text(res.get("output_text"))
        return as_text(res)

    def _translate_cloudflare(self, text, gender=None):
        import requests, json as _json
        acc, tok = self.cloudflare
        _, prompt = self._llm_prompt(text, gender)
        url = f"https://api.cloudflare.com/client/v4/accounts/{acc}/ai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {tok}"}
        # Disable model "thinking" for fast, non-empty replies
        extra = {} if getattr(self, "_cf_no_kwargs", False) else \
            {"chat_template_kwargs": {"enable_thinking": False}}
        for attempt in range(3):
            body = {"model": CLOUDFLARE_MODEL, "temperature": 0.5, "max_tokens": 512,
                    "messages": [{"role": "user", "content": prompt}], **extra}
            r = requests.post(url, headers=headers, json=body, timeout=30)
            if r.status_code == 400 and extra and \
                    any(w in r.text.lower() for w in ("chat_template_kwargs", "enable_thinking",
                                                     "additional properties", "unknown")):
                self._cf_no_kwargs, extra = True, {}
                continue
            if r.status_code in (429, 503) and attempt < 2:
                time.sleep(2)
                continue
            r.raise_for_status()
            data = r.json()
            out = self._clean_llm(self._extract_text(data))
            if out:
                return out
            print("CLOUDFLARE EMPTY RESPONSE:", _json.dumps(data, ensure_ascii=False)[:800], flush=True)
            raise RuntimeError("Cloudflare returned empty")
        r.raise_for_status()
        raise RuntimeError("Cloudflare not responding")

    def _translate_gemini(self, text, gender=None):
        import requests
        system, prompt = self._llm_prompt(text, gender)
        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 200,
                                 "thinkingConfig": {"thinkingBudget": 0}},
            "safetySettings": [
                {"category": c, "threshold": "BLOCK_NONE"} for c in (
                    "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT")
            ],
        }
        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
            params={"key": self.gemini_key}, json=body, timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        parts = data["candidates"][0].get("content", {}).get("parts", [])
        out = self._clean_llm("".join(p.get("text", "") for p in parts))
        if not out:
            raise RuntimeError("Gemini returned nothing (possibly blocked)")
        return out

    @staticmethod
    def _first_text(data):
        """Pull the translated string out of the many response shapes these endpoints use."""
        if isinstance(data, str):
            return data
        if isinstance(data, dict):
            if "translations" in data:
                return SubtitleEngine._first_text(data["translations"])
            if "sentences" in data:
                return "".join(s.get("trans", "") for s in data["sentences"])
            return data.get("text") or data.get("translatedText") or ""
        if isinstance(data, list) and data:
            if all(isinstance(x, str) for x in data):
                return data[0]
            return SubtitleEngine._first_text(data[0])
        return ""

    def _translate_google(self, text, target):
        """Google (Chrome dictionary endpoint, far less rate-limited than client=gtx)."""
        import requests
        r = requests.get(
            "https://clients5.google.com/translate_a/t",
            params={"client": "dict-chrome-ex", "sl": SOURCE_LANG, "tl": target, "q": text},
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/126.0"},
            timeout=8,
        )
        r.raise_for_status()
        out = self._first_text(r.json()).strip()
        if not out:
            raise RuntimeError("Google returned empty")
        return out

    def _translate_gtx(self, text, target):
        import requests
        r = requests.get(
            "https://translate.googleapis.com/translate_a/single",
            params={"client": "gtx", "sl": SOURCE_LANG, "tl": target, "dt": "t", "q": text},
            headers={"User-Agent": "Mozilla/5.0"}, timeout=8,
        )
        r.raise_for_status()
        out = "".join(part[0] for part in r.json()[0] if part and part[0])
        if not out:
            raise RuntimeError("Google returned empty")
        return out

    def _translate_edge(self, text, target):
        """Microsoft Translator via Edge's keyless endpoint (the old auth flow was retired)."""
        import requests
        r = requests.post(
            "https://edge.microsoft.com/translate/translatetext",
            params={"from": SOURCE_LANG, "to": target, "isEnterpriseClient": "false"},
            headers={"Content-Type": "application/json",
                     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Edg/126.0"},
            json=[text], timeout=8,
        )
        r.raise_for_status()
        try:
            data = r.json()
        except ValueError:
            data = r.text
        out = self._first_text(data).strip()
        if not out:
            raise RuntimeError("Microsoft returned empty")
        return out

    def _translate_mymemory(self, text, target):
        import requests
        r = requests.get("https://api.mymemory.translated.net/get",
                         params={"q": text, "langpair": f"{SOURCE_LANG}|{target}"}, timeout=8)
        r.raise_for_status()
        data = r.json()
        if int(data.get("responseStatus", 0)) != 200:
            raise RuntimeError(data.get("responseDetails", "error"))
        out = (data["responseData"]["translatedText"] or "").strip()
        if not out or out.startswith(("/", "[")) or "/(" in out:     # dictionary junk
            raise RuntimeError("MyMemory returned a dictionary entry")
        return out

    def machine_translate(self, text, target):
        """Try services in order; a blocked service (403/429) cools down for 5 min."""
        if not hasattr(self, "_cache"):
            self._cache, self._cooldown = {}, {}
        key = (text, target)
        if key in self._cache:
            return self._cache[key]
        services = [
            ("Microsoft", self._translate_edge),
            ("Google", self._translate_google),
            ("Google2", self._translate_gtx),
        ]
        if target != "en":           # MyMemory's English output is too unreliable
            services.append(("MyMemory", self._translate_mymemory))
        errors = []
        now = time.time()
        for name, fn in services:
            if self._cooldown.get(name, 0) > now:
                continue
            try:
                out = fn(text, target)
                self._cache[key] = out
                return out
            except Exception as e:
                code = getattr(getattr(e, "response", None), "status_code", None)
                if code in (403, 429):
                    self._cooldown[name] = now + 300      # rate limited: 5 min
                elif code in (404, 410):
                    self._cooldown[name] = now + 3600     # endpoint gone: 1 h
                errors.append(f"{name}: {code or type(e).__name__}")
        msg = ", ".join(errors) or "all services cooling down"
        if msg != getattr(self, "_last_mt_error", None):   # don't flood the log
            print("TRANSLATION ERROR:", target, msg, flush=True)
            self._last_mt_error = msg
        if target == "en" and TARGET_LANG != "en":
            return None              # English is only a helper line: just hide it
        return f"(translation failed – {msg})"

    def stop(self):
        self.running = False
        if self.capture:
            self.capture.stop()


# ---------------------------------------------------------------- UI
def run_ui():
    import json
    import tkinter as tk
    import tkinter.font as tkfont
    from tkinter import filedialog

    # Crisp text on scaled displays
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass

    # ------------------------------------------------------------ themes
    THEMES = {
        "dark": {
            "bg": "#101217", "surface": "#171a21", "hover": "#222631", "border": "#242833",
            "text": "#e9ebf0", "old": "#b9bdc7", "muted": "#7d8491", "faint": "#4b515d",
            "accent": "#ff7aa8", "en": "#98b6ff", "ok": "#4fd18b", "warn": "#f5b94c",
            "err": "#ff6b6b", "select": "#2c3342", "tip_bg": "#2a2f3a", "tip_fg": "#e9ebf0",
            "key": "#111318",   # transparency key (close to bg to avoid text fringes)
            "time": "#8a93a6", "sub": "#7cc4b0",
        },
        "light": {
            "bg": "#f7f7fa", "surface": "#ebecf1", "hover": "#dcdee6", "border": "#dfe1e7",
            "text": "#15171c", "old": "#4d535e", "muted": "#687080", "faint": "#a2a8b4",
            "accent": "#e0467c", "en": "#2d5bd0", "ok": "#1e9a5a", "warn": "#c2860f",
            "err": "#d43f3f", "select": "#cfd8ea", "tip_bg": "#23262e", "tip_fg": "#f2f3f6",
            "key": "#f6f7fa",
            "time": "#5d6574", "sub": "#17735f",
        },
    }
    C = {}
    painted = []   # (widget, {option: color key}) for theme switching

    def paint(w, **opts):
        painted.append((w, opts))
        w.config(**{k: C[v] for k, v in opts.items()})
        return w

    SETTINGS_FILE = os.path.join(APP_DIR, "japo_settings.json")
    state = {"size": 13, "show_ja": True, "show_en": True, "side": "right", "uncensored": False,
             "romaji": True, "gender": True, "theme": "dark", "alpha": 1.0, "width": 340, "topmost": True}
    try:
        with open(SETTINGS_FILE, encoding="utf-8") as f:
            state.update(json.load(f))
    except Exception:
        pass
    C.update(THEMES.get(state.get("theme"), THEMES["dark"]))

    def save_settings():
        try:
            state["width"] = int(root.winfo_width() / scale)
            if root.state() == "normal":          # skip when minimized/maximized
                state["geometry"] = root.geometry()   # "WxH+X+Y"
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=1)
        except Exception:
            pass

    ui_q = queue.Queue()
    engine = SubtitleEngine(ui_q)
    if "adult" in state:                       # settings from older versions
        state["uncensored"] = bool(state.pop("adult"))
    engine.uncensored = bool(state.get("uncensored"))
    engine.detect_gender = bool(state.get("gender", True))
    engine.movie_context = str(state.get("context", ""))
    history = []

    root = tk.Tk()
    root.title("Japo")
    paint(root, bg="bg")
    root.attributes("-topmost", state["topmost"])
    def set_window_icon():
        """Title bar / taskbar icon follows the theme (icon.ico = dark, icon_light.ico = light)."""
        name = "icon_light.ico" if state.get("theme") == "light" else "icon.ico"
        for base in (getattr(sys, "_MEIPASS", ""), APP_DIR):
            for fname in (name, "icon.ico"):
                ico = os.path.join(base, fname) if base else ""
                if ico and os.path.isfile(ico):
                    try:
                        root.iconbitmap(default=ico)
                    except Exception:
                        pass
                    return

    set_window_icon()

    scale = root.winfo_fpixels("1i") / 96.0
    px = lambda v: int(v * scale)
    root.minsize(px(260), px(320))

    families = set(tkfont.families())
    UI = next((f for f in ("Segoe UI Variable Text", "Segoe UI", "Inter", "DejaVu Sans")
               if f in families), "TkDefaultFont")
    UI_B = next((f for f in ("Segoe UI Variable Display", "Segoe UI", "Inter", "DejaVu Sans")
                 if f in families), UI)
    JP = next((f for f in ("Yu Gothic UI", "Meiryo UI", "Noto Sans CJK JP") if f in families), UI)
    # Icons: PNGs pre-rendered from SVG (japo_icons.py), sized for the display scale
    import japo_icons
    ICON_PX = min(japo_icons.SIZES, key=lambda s: abs(s - 20 * scale))
    _img_cache = {}

    def img(name):
        key = f"{name}|{state['theme']}|{ICON_PX}"
        if key not in _img_cache:
            _img_cache[key] = tk.PhotoImage(data=japo_icons.ICONS[key])
        return _img_cache[key]

    def dock():
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        w = max(px(260), px(state["width"]))
        h = sh - px(80)
        x = sw - w - px(8) if state["side"] == "right" else 0
        root.geometry(f"{w}x{h}+{x}+0")

    def screen_area():
        """Virtual screen bounds (all monitors)."""
        if os.name == "nt":
            try:
                import ctypes
                gm = ctypes.windll.user32.GetSystemMetrics
                return gm(76), gm(77), gm(78), gm(79)   # x, y, width, height
            except Exception:
                pass
        return 0, 0, root.winfo_screenwidth(), root.winfo_screenheight()

    def style_titlebar():
        """Match the Windows title bar to the theme.
        Win10: dark/light mode. Win11: caption, text and border colors too."""
        if os.name != "nt":
            return
        try:
            import ctypes
            from ctypes import wintypes, byref, sizeof, c_int
            root.update_idletasks()
            hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
            dwm = ctypes.windll.dwmapi.DwmSetWindowAttribute
            dark = c_int(1 if state["theme"] == "dark" else 0)
            for attr in (20, 19):          # DWMWA_USE_IMMERSIVE_DARK_MODE (new / old id)
                if dwm(hwnd, attr, byref(dark), sizeof(dark)) == 0:
                    break

            def colorref(hex_color):
                r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
                return wintypes.DWORD(r | (g << 8) | (b << 16))

            # Win11: caption color = toolbar color
            for attr, key in ((35, "surface"), (36, "text"), (34, "border")):
                col = colorref(THEMES[state["theme"]][key])
                dwm(hwnd, attr, byref(col), sizeof(col))
            # Redraw the frame now
            ctypes.windll.user32.SetWindowPos(hwnd, 0, 0, 0, 0, 0, 0x0001 | 0x0002 | 0x0004 | 0x0010 | 0x0020)
        except Exception as e:
            print("Titlebar theme not applied:", e, flush=True)

    def restore_position():
        """Reopen at the last position; fall back to docking if it is off-screen."""
        import re
        m = re.fullmatch(r"(\d+)x(\d+)\+(-?\d+)\+(-?\d+)", str(state.get("geometry") or ""))
        if m:
            w, h, x, y = map(int, m.groups())
            vx, vy, vw, vh = screen_area()
            if (w >= px(200) and h >= px(200) and vx - w + px(80) <= x <= vx + vw - px(80)
                    and vy <= y <= vy + vh - px(60)):
                root.geometry(f"{w}x{h}+{x}+{y}")
                return
        dock()

    restore_position()

    # ------------------------------------------------------------ tooltip
    class Tip:
        def __init__(self, widget, text):
            self.w, self.text, self.win, self.job = widget, text, None, None
            widget.bind("<Enter>", self.schedule, add="+")
            widget.bind("<Leave>", self.hide, add="+")
            widget.bind("<ButtonPress>", self.hide, add="+")

        def schedule(self, _=None):
            self.job = self.w.after(450, self.show)

        def show(self):
            if self.win:
                return
            self.win = tk.Toplevel(self.w)
            self.win.wm_overrideredirect(True)
            self.win.attributes("-topmost", True)
            tk.Label(self.win, text=self.text() if callable(self.text) else self.text,
                     bg=C["tip_bg"], fg=C["tip_fg"], font=(UI, 9), padx=8, pady=4).pack()
            self.win.update_idletasks()
            x = self.w.winfo_rootx() + self.w.winfo_width() // 2 - self.win.winfo_width() // 2
            x = max(0, min(x, root.winfo_screenwidth() - self.win.winfo_width()))
            self.win.geometry(f"+{x}+{self.w.winfo_rooty() + self.w.winfo_height() + px(4)}")

        def hide(self, _=None):
            if self.job:
                self.w.after_cancel(self.job)
                self.job = None
            if self.win:
                self.win.destroy()
                self.win = None

    # ------------------------------------------------------------ flat buttons
    icon_buttons = []

    class FlatButton(tk.Label):
        """Flat button with an icon (from japo_icons) or short text."""
        def __init__(self, parent, command, icon=None, text=None, tip=None, toggle=None):
            if icon:
                super().__init__(parent, image=img(icon), padx=px(7), pady=px(5), cursor="hand2")
                icon_buttons.append(self)
            else:
                super().__init__(parent, text=text, font=(UI, 9, "bold"),
                                 padx=px(9), pady=px(5), cursor="hand2")
            paint(self, bg="surface", fg="text")
            self.icon = icon
            self.command, self.toggle = command, toggle
            self.bind("<Enter>", lambda e: self.config(bg=C["hover"]))
            self.bind("<Leave>", lambda e: self.config(bg=C["surface"]))
            self.bind("<ButtonRelease-1>", lambda e: self.command())
            if tip:
                Tip(self, tip)
            self.refresh()

        def set_icon(self, name):
            self.icon = name
            self.config(image=img(name))

        def refresh(self):
            if self.icon:
                self.config(image=img(self.icon))
            if self.toggle is not None:
                self.config(fg=C["text"] if self.toggle() else C["faint"])

    # ============================================================ TOPBAR
    # Toolbar flush with the top edge
    toolbar = paint(tk.Frame(root), bg="surface")
    toolbar.pack(side="top", fill="x")
    inner = paint(tk.Frame(toolbar), bg="surface")
    inner.pack(fill="x", padx=px(8), pady=px(5))
    paint(tk.Frame(root, height=1), bg="border").pack(side="top", fill="x")

    # ============================================================ FILM CONTEXT
    PLACEHOLDER = "Film title, plot or characters…"
    ctx_bar = paint(tk.Frame(root), bg="bg")
    ctx_bar.pack(side="top", fill="x", padx=px(12), pady=(px(8), px(2)))
    ctx_entry = paint(tk.Entry(ctx_bar, font=(UI, 9), relief="flat", bd=0,
                               highlightthickness=1),
                      bg="surface", insertbackground="text",
                      highlightbackground="border", highlightcolor="accent")
    ctx_entry.pack(fill="x", ipady=px(5), ipadx=px(6))
    ctx_state = {"placeholder": False, "job": None}

    def show_placeholder():
        if not ctx_entry.get() and root.focus_get() is not ctx_entry:
            ctx_entry.insert(0, PLACEHOLDER)
            ctx_state["placeholder"] = True
        ctx_entry.config(fg=C["muted"] if ctx_state["placeholder"] else C["text"])

    def ctx_value():
        return "" if ctx_state["placeholder"] else ctx_entry.get().strip()

    def commit_context(_=None):
        ctx_state["job"] = None
        value = ctx_value()
        if value != engine.movie_context:
            engine.movie_context = value
            state["context"] = value
            save_settings()

    def on_ctx_focus_in(_):
        if ctx_state["placeholder"]:
            ctx_entry.delete(0, "end")
            ctx_state["placeholder"] = False
            ctx_entry.config(fg=C["text"])

    def on_ctx_focus_out(_):
        commit_context()
        show_placeholder()

    def on_ctx_key(_):
        if ctx_state["job"]:
            root.after_cancel(ctx_state["job"])
        ctx_state["job"] = root.after(800, commit_context)   # save shortly after typing stops

    if engine.movie_context:
        ctx_entry.insert(0, engine.movie_context)
    ctx_entry.bind("<FocusIn>", on_ctx_focus_in)
    ctx_entry.bind("<FocusOut>", on_ctx_focus_out)
    ctx_entry.bind("<KeyRelease>", on_ctx_key)
    ctx_entry.bind("<Return>", lambda e: root.focus_set())
    ctx_entry.bind("<Escape>", lambda e: root.focus_set())
    Tip(ctx_entry, "Helps the AI with names, relationships and forms of address")
    show_placeholder()

    def set_font_size(delta):
        state["size"] = min(28, max(9, state["size"] + delta))
        apply_fonts()
        save_settings()

    def toggle_state(key):
        state[key] = not state[key]
        apply_fonts()
        for b in toggles:
            b.refresh()
        save_settings()

    def toggle_pause():
        engine.paused = not engine.paused
        pause_btn.set_icon("play" if engine.paused else "pause")

    def clear():
        history.clear()
        text.config(state="normal")
        text.delete("1.0", "end")
        text.config(state="disabled")
        show_empty(True)

    def vsep():
        paint(tk.Frame(inner, width=1), bg="border").pack(side="left", fill="y",
                                                          padx=px(8), pady=px(6))

    # left: playback + display toggles
    pause_btn = FlatButton(inner, toggle_pause, icon="pause",
                           tip=lambda: "Resume" if engine.paused else "Pause")
    pause_btn.pack(side="left", padx=px(2))
    FlatButton(inner, clear, icon="clear", tip="Clear all").pack(side="left", padx=px(2))
    vsep()
    ja_btn = FlatButton(inner, lambda: toggle_state("show_ja"), text="JP",
                        tip="Show/hide Japanese", toggle=lambda: state["show_ja"])
    en_btn = FlatButton(inner, lambda: toggle_state("show_en"), text="EN",
                        tip="Show/hide English", toggle=lambda: state["show_en"])
    ja_btn.pack(side="left", padx=px(2))
    en_btn.pack(side="left", padx=px(2))
    toggles = [ja_btn, en_btn]

    # right: text size, theme, settings
    more_btn = FlatButton(inner, lambda: open_menu(), icon="settings", tip="Settings")
    more_btn.pack(side="right", padx=px(2))
    theme_btn = FlatButton(inner, lambda: toggle_theme(),
                           icon="sun" if state["theme"] == "dark" else "moon",
                           tip=lambda: "Light theme" if state["theme"] == "dark" else "Dark theme")
    theme_btn.pack(side="right", padx=px(2))
    paint(tk.Frame(inner, width=1), bg="border").pack(side="right", fill="y",
                                                      padx=px(8), pady=px(6))
    FlatButton(inner, lambda: set_font_size(1), icon="font_up", tip="Larger text"
               ).pack(side="right", padx=px(2))
    FlatButton(inner, lambda: set_font_size(-1), icon="font_dn", tip="Smaller text"
               ).pack(side="right", padx=px(2))

    # Minimum window width = everything on the toolbar fits (plus a small gap in the middle)
    def fit_min_width():
        root.update_idletasks()
        need = inner.winfo_reqwidth() + px(8) * 2 + px(16)
        root.minsize(need, px(320))
        if root.winfo_width() < need:
            root.geometry(f"{need}x{root.winfo_height()}")
    root.after(50, fit_min_width)

    # ============================================================ STATUS BAR
    footer = paint(tk.Frame(root), bg="bg")
    footer.pack(side="bottom", fill="x", padx=px(12), pady=(px(5), px(7)))
    paint(tk.Frame(root, height=1), bg="border").pack(side="bottom", fill="x")

    # Meter and dot stay fixed on the left so changing status text never shifts them
    meter = paint(tk.Canvas(footer, width=px(26), height=px(12), highlightthickness=0), bg="bg")
    meter.pack(side="left", padx=(0, px(8)))
    bars = [meter.create_rectangle(0, 0, 0, 0, fill=C["faint"], outline="") for _ in range(5)]
    Tip(meter, "Audio level")

    dot = paint(tk.Canvas(footer, width=px(8), height=px(8), highlightthickness=0), bg="bg")
    dot_id = dot.create_oval(0, 0, px(8), px(8), fill=C["warn"], outline="")
    dot.pack(side="left", padx=(0, px(6)))
    pill_text = paint(tk.Label(footer, text="Starting", font=(UI, 8, "bold")), bg="bg", fg="text")
    pill_text.pack(side="left")

    foot_right = paint(tk.Label(footer, text="", font=(UI, 8), anchor="e"), bg="bg", fg="muted")
    foot_right.pack(side="right")
    note = paint(tk.Label(root, text="", font=(UI, 8), anchor="w", justify="left"),
                 bg="bg", fg="warn")

    # ============================================================ SUBTITLE LIST
    body = paint(tk.Frame(root), bg="bg")
    body.pack(side="top", fill="both", expand=True, padx=(px(12), px(4)), pady=(px(4), 0))

    text = paint(tk.Text(body, wrap="word", bd=0, highlightthickness=0,
                         padx=0, pady=px(2), cursor="arrow", width=1, height=1, insertwidth=0),
                 bg="bg", fg="text", selectbackground="select", selectforeground="text")
    text.pack(side="left", fill="both", expand=True)

    # Thin custom scrollbar (native one cannot be themed)
    sb = paint(tk.Canvas(body, width=px(6), highlightthickness=0), bg="bg")
    sb.pack(side="right", fill="y", padx=(px(4), 0))
    thumb = sb.create_rectangle(0, 0, 0, 0, fill=C["border"], outline="")

    def on_scroll(first, last):
        first, last = float(first), float(last)
        h = sb.winfo_height()
        if last - first >= 0.999:
            sb.coords(thumb, 0, 0, 0, 0)
        else:
            sb.coords(thumb, px(1), first * h, px(5), max(last * h, first * h + px(20)))
    text.config(yscrollcommand=on_scroll)

    drag = {"y": None}
    sb.bind("<ButtonPress-1>", lambda e: drag.update(y=e.y))
    def on_drag(e):
        if drag["y"] is None:
            return
        frac = (e.y - drag["y"]) / max(1, sb.winfo_height())
        drag["y"] = e.y
        text.yview_moveto(text.yview()[0] + frac)
    sb.bind("<B1-Motion>", on_drag)
    sb.bind("<Enter>", lambda e: sb.itemconfig(thumb, fill=C["faint"]))
    sb.bind("<Leave>", lambda e: sb.itemconfig(thumb, fill=C["border"]))

    empty = paint(tk.Label(text, text="Waiting for dialogue…\nPlay a video and subtitles appear here.",
                           font=(UI, 10), justify="center"), bg="bg", fg="faint")

    def show_empty(on):
        if on:
            empty.place(relx=0.5, rely=0.4, anchor="center")
        else:
            empty.place_forget()

    show_empty(True)

    def apply_fonts():
        s = state["size"]
        text.tag_configure("time", foreground=C["time"], font=(UI, max(8, s - 4)),
                           spacing1=px(10))
        text.tag_configure("ja", foreground=C["sub"], font=(JP, max(8, s - 3)),
                           elide=not state["show_ja"], spacing1=px(2))
        use_ro = state.get("romaji", True)
        text.tag_configure("ro", foreground=C["sub"], font=(UI, max(8, s - 3)),
                           elide=not (state["show_ja"] and use_ro), spacing1=px(2))
        text.tag_configure("ja_alt", foreground=C["sub"], font=(JP, max(8, s - 3)),
                           elide=not (state["show_ja"] and not use_ro), spacing1=px(2))
        text.tag_configure("en", foreground=C["en"], font=(UI, max(8, s - 3), "italic"),
                           elide=not state["show_en"], spacing1=px(2))
        text.tag_configure("vi", foreground=C["text"], font=(UI_B, s, "bold"),
                           spacing1=px(3), spacing3=px(10))
        text.tag_configure("vi_old", foreground=C["old"], font=(UI_B, s, "bold"))
        text.tag_configure("rule", font=(UI, 2), background=C["border"])
        text.tag_configure("err", foreground=C["err"], font=(UI, max(8, s - 4)),
                           spacing1=px(6), spacing3=px(6))
        text.tag_configure("g_f", foreground=C["accent"])
        text.tag_configure("g_m", foreground=C["en"])
        text.tag_raise("g_f")
        text.tag_raise("g_m")
        text.tag_raise("vi_old")
    apply_fonts()
    text.config(state="disabled")

    last_vi = {"start": None}

    def add_line(ts, ja, vi, en, ro="", gender=None):
        show_empty(False)
        at_bottom = text.yview()[1] > 0.98
        text.config(state="normal")
        if last_vi["start"]:
            text.tag_add("vi_old", last_vi["start"], "vi_end")
        text.insert("end", f"{ts:%H:%M:%S}", "time")
        if gender in ("f", "m"):
            text.insert("end", "  ♀" if gender == "f" else "  ♂", ("time", "g_" + gender))
        text.insert("end", "\n", "time")
        # Insert both romaji and Japanese; visibility is toggled via tags
        if ro:
            text.insert("end", ro + "\n", "ro")
            text.insert("end", ja + "\n", "ja_alt")
        else:
            text.insert("end", ja + "\n", "ja")
        if en:
            text.insert("end", en + "\n", "en")
        last_vi["start"] = text.index("end-1c")
        text.insert("end", vi + "\n", "vi")
        text.mark_set("vi_end", "end-1c")
        text.mark_gravity("vi_end", "left")
        text.config(state="disabled")
        if at_bottom:
            text.see("end")

    def add_error(msg):
        show_empty(False)
        text.config(state="normal")
        text.insert("end", f"⚠ {msg}\n", "err")
        text.config(state="disabled")
        text.see("end")

    # ============================================================ MENU
    MENU_COLORS = dict(bg="surface", fg="text", activebackground="hover", activeforeground="text",
                       selectcolor="text")
    menu = paint(tk.Menu(root, tearoff=0, bd=0, font=(UI, 9)), **MENU_COLORS)
    top_var = tk.BooleanVar(value=state["topmost"])
    alpha_var = tk.DoubleVar(value=state["alpha"])

    def switch_side():
        state["side"] = "left" if state["side"] == "right" else "right"
        state["width"] = int(root.winfo_width() / scale)
        dock()
        save_settings()

    # ---- Background-only transparency:
    # the main window keys out its bg color (-transparentcolor) over a translucent backdrop (-alpha)
    backdrop = tk.Toplevel(root)
    backdrop.overrideredirect(True)
    backdrop.withdraw()
    bd_state = {"on": False}

    def sync_backdrop(_=None):
        if not bd_state["on"]:
            return
        try:
            x, y = root.winfo_rootx(), root.winfo_rooty()
            w, h = root.winfo_width(), root.winfo_height()
            backdrop.geometry(f"{w}x{h}+{x}+{y}")
        except tk.TclError:
            pass

    def raise_pair(_=None):
        if bd_state["on"]:
            backdrop.lift()
            root.lift()

    def on_unmap(_=None):
        if bd_state["on"] and root.state() == "iconic":
            backdrop.withdraw()

    def on_map(_=None):
        if bd_state["on"]:
            backdrop.deiconify()
            sync_backdrop()
            raise_pair()

    root.bind("<Configure>", lambda e: root.after_idle(sync_backdrop), add="+")
    root.bind("<FocusIn>", raise_pair, add="+")
    root.bind("<Unmap>", on_unmap, add="+")
    root.bind("<Map>", on_map, add="+")
    # Clicks/scrolls on the transparent area go back to the main window
    backdrop.bind("<Button-1>", lambda e: (root.focus_force(), raise_pair()))
    backdrop.bind("<MouseWheel>", lambda e: text.yview_scroll(int(-e.delta / 120) * 2, "units"))

    def apply_opacity():
        a = state["alpha"]
        on = a < 0.999
        base = THEMES[state["theme"]]
        C["bg"] = base["key"] if on else base["bg"]
        try:
            root.attributes("-transparentcolor", C["bg"] if on else "")
        except tk.TclError:
            on = False            # unsupported on this OS
            C["bg"] = base["bg"]
        bd_state["on"] = on
        if on:
            backdrop.config(bg=base["bg"])
            backdrop.attributes("-alpha", a)
            backdrop.attributes("-topmost", state["topmost"])
            backdrop.deiconify()
            sync_backdrop()
            raise_pair()
        else:
            backdrop.withdraw()
        repaint()

    def set_alpha():
        state["alpha"] = alpha_var.get()
        apply_opacity()
        save_settings()

    def set_top():
        state["topmost"] = top_var.get()
        root.attributes("-topmost", state["topmost"])
        backdrop.attributes("-topmost", state["topmost"])
        raise_pair()
        save_settings()

    def save_txt():
        path = filedialog.asksaveasfilename(defaultextension=".txt", initialfile="japo_subtitles.txt",
                                            filetypes=[("Text", "*.txt")])
        if path:
            with open(path, "w", encoding="utf-8") as f:
                if engine.movie_context:
                    f.write(f"Film context: {engine.movie_context}\n\n")
                for ts, ja, vi, en, ro, g in history:
                    tag = {"f": " (F)", "m": " (M)"}.get(g, "")
                    f.write(f"[{ts:%H:%M:%S}]{tag}\n{ja}\n" + (f"{ro}\n" if ro else "")
                            + (f"{en}\n" if en else "") + f"{vi}\n\n")

    alpha_menu = paint(tk.Menu(menu, tearoff=0, bd=0, font=(UI, 9)), **MENU_COLORS)
    for label, val in (("100%", 1.0), ("80%", 0.8), ("60%", 0.6), ("40%", 0.4), ("20%", 0.2)):
        alpha_menu.add_radiobutton(label=label, value=val, variable=alpha_var, command=set_alpha)

    light_var = tk.BooleanVar(value=state["theme"] == "light")
    menu.add_checkbutton(label="Light theme", variable=light_var,
                         command=lambda: toggle_theme())
    menu.add_command(label="Dock left / right", command=switch_side)
    menu.add_cascade(label="Background opacity", menu=alpha_menu)
    menu.add_checkbutton(label="Always on top", variable=top_var, command=set_top)
    menu.add_separator()
    uncensored_var = tk.BooleanVar(value=bool(state.get("uncensored")))

    def set_uncensored():
        state["uncensored"] = engine.uncensored = uncensored_var.get()
        save_settings()

    menu.add_checkbutton(label="Uncensored translation", variable=uncensored_var, command=set_uncensored)
    ro_var = tk.BooleanVar(value=bool(state.get("romaji", True)))

    def set_romaji():
        state["romaji"] = ro_var.get()
        apply_fonts()
        save_settings()

    menu.add_checkbutton(label="Show romaji", variable=ro_var, command=set_romaji)
    gender_var = tk.BooleanVar(value=bool(state.get("gender", True)))

    def set_gender():
        state["gender"] = engine.detect_gender = gender_var.get()
        save_settings()

    menu.add_checkbutton(label="Detect speaker gender", variable=gender_var, command=set_gender)
    menu.add_separator()
    # ---------------------------------------------------------------- API keys dialog
    def open_api_dialog():
        if getattr(open_api_dialog, "win", None) and open_api_dialog.win.winfo_exists():
            open_api_dialog.win.lift()
            return
        cf = read_cloudflare() or ("", "")
        gm = read_gemini_key() or ""

        win = tk.Toplevel(root)
        open_api_dialog.win = win
        win.title("API keys")
        win.configure(bg=C["bg"])
        win.resizable(False, False)
        win.transient(root)
        win.attributes("-topmost", True)
        pad = px(16)

        def label(parent, text, small=False):
            return tk.Label(parent, text=text, bg=C["bg"], fg=C["muted"] if small else C["text"],
                            font=(UI, 8 if small else 9, "normal" if small else "bold"),
                            anchor="w", justify="left")

        def entry(parent, value, secret=False):
            e = tk.Entry(parent, font=(UI, 9), relief="flat", bd=0, highlightthickness=1,
                         bg=C["surface"], fg=C["text"], insertbackground=C["text"],
                         highlightbackground=C["border"], highlightcolor=C["accent"],
                         show="•" if secret else "", width=42)
            e.insert(0, value)
            return e

        body_f = tk.Frame(win, bg=C["bg"])
        body_f.pack(fill="both", padx=pad, pady=(pad, px(8)))

        label(body_f, "Cloudflare Workers AI").pack(fill="x")
        label(body_f, "Used for AI translation (Gemma). Create a token with the “Workers AI” template.",
              small=True).pack(fill="x", pady=(0, px(8)))

        label(body_f, "Account ID", small=True).pack(fill="x")
        acc_e = entry(body_f, cf[0])
        acc_e.pack(fill="x", ipady=px(4), pady=(px(2), px(8)))

        label(body_f, "API token", small=True).pack(fill="x")
        tok_row = tk.Frame(body_f, bg=C["bg"])
        tok_row.pack(fill="x", pady=(px(2), px(4)))
        tok_e = entry(tok_row, cf[1], secret=True)
        tok_e.pack(side="left", fill="x", expand=True, ipady=px(4))

        def toggle_show():
            tok_e.config(show="" if tok_e.cget("show") else "•")
            gem_e.config(show="" if gem_e.cget("show") else "•")
        show_btn = tk.Label(tok_row, text="Show", bg=C["surface"], fg=C["text"], font=(UI, 8),
                            padx=px(8), pady=px(4), cursor="hand2")
        show_btn.pack(side="left", padx=(px(6), 0))
        show_btn.bind("<Button-1>", lambda e: toggle_show())

        link = tk.Label(body_f, text="Get a token →", bg=C["bg"], fg=C["en"], font=(UI, 8, "underline"),
                        cursor="hand2", anchor="w")
        link.pack(fill="x", pady=(0, px(12)))
        link.bind("<Button-1>", lambda e: __import__("webbrowser").open(
            "https://dash.cloudflare.com/profile/api-tokens"))

        label(body_f, "Gemini (optional)").pack(fill="x")
        label(body_f, "Fallback when Cloudflare is unavailable.", small=True).pack(fill="x", pady=(0, px(6)))
        gem_e = entry(body_f, gm, secret=True)
        gem_e.pack(fill="x", ipady=px(4), pady=(0, px(4)))

        status_l = label(body_f, "", small=True)
        status_l.pack(fill="x", pady=(px(8), 0))

        btns = tk.Frame(win, bg=C["bg"])
        btns.pack(fill="x", padx=pad, pady=(0, pad))

        def mk_btn(text, cmd, primary=False):
            b = tk.Label(btns, text=text, font=(UI, 9, "bold" if primary else "normal"),
                         bg=C["accent"] if primary else C["surface"],
                         fg="#ffffff" if primary else C["text"],
                         padx=px(14), pady=px(6), cursor="hand2")
            b.bind("<Button-1>", lambda e: cmd())
            return b

        def write_or_remove(path, content):
            if content.strip():
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content.strip() + "\n")
            elif os.path.exists(path):
                os.remove(path)

        def save():
            acc, tok, gem = acc_e.get().strip(), tok_e.get().strip(), gem_e.get().strip()
            if tok.lower().startswith("bearer "):
                tok = tok[7:].strip()
            if bool(acc) != bool(tok):
                status_l.config(text="Please fill in both Account ID and API token.", fg=C["warn"])
                return
            try:
                write_or_remove(os.path.join(APP_DIR, "cloudflare.txt"), f"{acc}\n{tok}" if acc else "")
                write_or_remove(os.path.join(APP_DIR, "gemini_key.txt"), gem)
            except OSError as e:
                status_l.config(text=f"Could not save: {e}", fg=C["err"])
                return
            engine.cloudflare = read_cloudflare()
            engine.gemini_key = read_gemini_key()
            engine.gemini_note = None
            win.destroy()

        def test():
            acc, tok = acc_e.get().strip(), tok_e.get().strip()
            if not tok:
                status_l.config(text="Enter an API token first.", fg=C["warn"])
                return
            status_l.config(text="Testing…", fg=C["muted"])

            def work():
                try:
                    import requests
                    r = requests.get("https://api.cloudflare.com/client/v4/user/tokens/verify",
                                     headers={"Authorization": f"Bearer {tok}"}, timeout=10)
                    ok = r.ok and r.json().get("result", {}).get("status") == "active"
                    msg = ("Token is valid ✓", C["ok"]) if ok else ("Invalid token", C["err"])
                except Exception as e:
                    msg = (f"Could not connect: {type(e).__name__}", C["err"])
                root.after(0, lambda: status_l.winfo_exists() and status_l.config(text=msg[0], fg=msg[1]))
            threading.Thread(target=work, daemon=True).start()

        mk_btn("Save", save, primary=True).pack(side="right")
        mk_btn("Cancel", win.destroy).pack(side="right", padx=(0, px(8)))
        mk_btn("Test", test).pack(side="left")

        win.bind("<Return>", lambda e: save())
        win.bind("<Escape>", lambda e: win.destroy())
        win.update_idletasks()
        x = root.winfo_rootx() + (root.winfo_width() - win.winfo_width()) // 2
        y = root.winfo_rooty() + px(60)
        x = min(x, root.winfo_screenwidth() - win.winfo_width() - px(8))
        win.geometry(f"+{max(0, x)}+{max(0, y)}")
        acc_e.focus_set()

    menu.add_command(label="API keys…", command=open_api_dialog)
    menu.add_command(label="Save subtitles (.txt)…", command=save_txt)
    menu.add_command(label="Open Japo folder", command=lambda: os.startfile(APP_DIR)
                     if os.name == "nt" else None)

    def toggle_theme():
        state["theme"] = "light" if state["theme"] == "dark" else "dark"
        light_var.set(state["theme"] == "light")
        C.update(THEMES[state["theme"]])
        apply_opacity()
        style_titlebar()
        set_window_icon()
        save_settings()

    def repaint():
        for w, opts in painted:
            try:
                w.config(**{k: C[v] for k, v in opts.items()})
            except tk.TclError:
                pass
        for b in toggles:
            b.refresh()
        theme_btn.icon = "sun" if state["theme"] == "dark" else "moon"
        for b in icon_buttons:
            b.refresh()
        sb.itemconfig(thumb, fill=C["border"])
        show_placeholder()
        apply_fonts()

    def open_menu():
        menu.tk_popup(more_btn.winfo_rootx() + more_btn.winfo_width() - px(200),
                      more_btn.winfo_rooty() + more_btn.winfo_height() + px(2))

    # ============================================================ UPDATE LOOP
    base_status = {"text": "Starting…"}

    def set_pill(color, label):
        dot.itemconfig(dot_id, fill=color)
        pill_text.config(text=label)

    def update_meter():
        lv = engine.level if engine.listening and not engine.paused else 0
        n = 0 if lv <= 0.02 else min(5, int(lv * 5) + 1)
        for i, b in enumerate(bars):
            h = px(3 + i * 2.2)
            meter.coords(b, i * px(5.5), px(12) - h, i * px(5.5) + px(3.5), px(12))
            meter.itemconfig(b, fill=C["accent"] if i < n else C["faint"])

    def poll():
        try:
            while True:
                msg = ui_q.get_nowait()
                if msg[0] == "status":
                    base_status["text"] = msg[1]
                elif msg[0] == "error":
                    print("ERROR:", msg[1], flush=True)
                    add_error(msg[1])
                elif msg[0] == "line":
                    _, ts, ja, vi, en, ro, g = msg
                    history.append((ts, ja, vi, en, ro, g))
                    add_line(ts, ja, vi, en, ro, g)
        except queue.Empty:
            pass

        # status indicator
        if not engine.listening:
            failed = engine.failed
            set_pill(C["err"] if failed else C["warn"], "Error" if failed else "Preparing")
            foot_right.config(text=base_status["text"])
        else:
            if engine.paused:
                set_pill(C["muted"], "Paused")
            elif engine.busy or engine.seg_q.qsize() or engine.tr_q.qsize():
                set_pill(C["accent"], "Translating")
            else:
                set_pill(C["ok"], "Listening")
            wait = engine.seg_q.qsize() + engine.tr_q.qsize()
            n = len(history)
            count = f"{wait} queued" if wait else f"{n} line{'s' if n != 1 else ''}"
            foot_right.config(text=f"{count}  ·  {'GPU' if engine.device == 'cuda' else 'CPU'}")

        note_text = engine.gemini_note
        if state.get("romaji", True) and _ROMAJI["error"]:
            note_text = (note_text + "\n" if note_text else "") + _ROMAJI["error"]
        if note_text:
            note.config(text=note_text, wraplength=max(100, root.winfo_width() - px(24)))
            if not note.winfo_ismapped():
                note.pack(side="bottom", fill="x", padx=px(12), after=footer)
        elif note.winfo_ismapped():
            note.pack_forget()

        update_meter()
        root.after(100, poll)

    def on_close():
        save_settings()
        engine.stop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.after(10, style_titlebar)
    if state["alpha"] < 0.999:
        root.after(300, apply_opacity)
    engine.start()
    poll()
    root.mainloop()


if __name__ == "__main__":
    run_ui()

"""
Japo video mode: replay a screen region and one app's audio with a delay, so
translated subtitles can be shown exactly when each line is spoken.

    browser ──audio──► AppAudio ──► MediaStore (disk) ──► speakers (delayed)
            │                  └──► SubtitleEngine (Whisper + translation)
            └──screen─► ScreenGrabber ──► MediaStore ──► Japo window (delayed)

Everything is kept for a while (default 15 min), so playback can be rewound.

Playback runs `delay` seconds behind real time. If the next spoken line is not
translated yet when playback reaches it, playback holds (up to MAX_WAIT s).
"""
import io
import queue
import threading
import time

import numpy as np

RATE, CHANNELS = 48000, 2
APP_VOLUME = 0.05          # browser volume while video mode runs (5 % stays clean when boosted)
MAX_WAIT = 5.0             # longest hold for a missing translation, in seconds
BLOCK = 1024               # audio output block (≈21 ms)


# ---------------------------------------------------------------- recording (for rewinding)
class MediaStore:
    """Everything captured in video mode, kept on disk for `keep_sec` seconds so
    playback can be rewound. Files are written in one-minute chunks to a temp folder.

    Audio: int16 stereo on the wall clock, sample i is at base + i / RATE
    (gaps where the app played nothing are filled with silence).
    Video: JPEG frames indexed by capture time."""

    CHUNK_SEC = 60

    def __init__(self, keep_sec=900):
        import os
        import tempfile
        self.keep_sec = keep_sec
        self.dir = tempfile.mkdtemp(prefix="japo_video_")
        self.os = os
        self.lock = threading.Lock()
        # audio
        self.base = None
        self.n = 0                       # frames written since base
        self.a_chunks = []               # [n0, n1, path]
        self._a_file = None
        # video
        self.f_times = []                # capture times (sorted) for bisect
        self.f_index = []                # (chunk path, offset, length)
        self._v_file = None
        self._v_path = None
        self._v_started = 0.0
        self._readers = {}               # path -> open read handle

    # ---- audio
    def add_audio(self, t_end, pcm16):
        """pcm16: int16 array (frames, 2) that ended at wall time t_end."""
        frames = len(pcm16)
        with self.lock:
            if self.base is None:
                self.base = t_end - frames / RATE
            gap = (t_end - frames / RATE) - (self.base + self.n / RATE)
            if gap > 0.08:                                   # missing audio -> silence
                self._write_audio(np.zeros((int(gap * RATE), CHANNELS), np.int16))
            elif gap < -0.5:                                 # clock drift: re-anchor
                self.base += gap
            self._write_audio(pcm16)

    def _write_audio(self, arr):
        if not self.a_chunks or self.a_chunks[-1][1] - self.a_chunks[-1][0] >= self.CHUNK_SEC * RATE:
            if self._a_file:
                self._a_file.close()
            path = self.os.path.join(self.dir, f"a{len(self.a_chunks):05d}.raw")
            self._a_file = open(path, "wb")
            self.a_chunks.append([self.n, self.n, path])
        self._a_file.write(arr.tobytes())
        self._a_file.flush()
        self.n += len(arr)
        self.a_chunks[-1][1] = self.n

    def read_audio(self, t, frames):
        """float32 (frames, 2) starting at wall time t (silence outside the recording)."""
        out = np.zeros((frames, CHANNELS), np.float32)
        with self.lock:
            if self.base is None:
                return out
            start = int(round((t - self.base) * RATE))
            end = start + frames
            for n0, n1, path in self.a_chunks:
                if n1 <= start or n0 >= end:
                    continue
                a, b = max(start, n0), min(end, n1)
                fh = self._reader(path)
                fh.seek((a - n0) * CHANNELS * 2)
                data = np.frombuffer(fh.read((b - a) * CHANNELS * 2), np.int16)
                got = len(data) // CHANNELS
                out[a - start:a - start + got] = data[:got * CHANNELS].reshape(-1, CHANNELS) / 32767.0
        return out

    # ---- video
    def add_frame(self, t, jpeg):
        with self.lock:
            if self._v_file is None or t - self._v_started >= self.CHUNK_SEC:
                if self._v_file:
                    self._v_file.close()
                self._v_path = self.os.path.join(self.dir, f"v{int(t * 1000)}.jpgs")
                self._v_file = open(self._v_path, "wb")
                self._v_started = t
            off = self._v_file.tell()
            self._v_file.write(jpeg)
            self._v_file.flush()
            if self.f_times and t < self.f_times[-1]:
                return                                       # keep the index sorted
            self.f_times.append(t)
            self.f_index.append((self._v_path, off, len(jpeg)))
            self._cleanup(t)

    def frame_at(self, t):
        """(capture time, jpeg bytes) of the latest frame at or before t, or None."""
        import bisect
        with self.lock:
            i = bisect.bisect_right(self.f_times, t) - 1
            if i < 0:
                return None
            path, off, ln = self.f_index[i]
            ft = self.f_times[i]
            fh = self._reader(path)
            fh.seek(off)
            return ft, fh.read(ln)

    # ---- range / housekeeping
    def oldest(self):
        with self.lock:
            cands = []
            if self.f_times:
                cands.append(self.f_times[0])
            if self.base is not None and self.a_chunks:
                cands.append(self.base + self.a_chunks[0][0] / RATE)
            return max(cands) if cands else None

    def _reader(self, path):
        fh = self._readers.get(path)
        if fh is None:
            fh = self._readers[path] = open(path, "rb")
        return fh

    def _drop(self, path):
        fh = self._readers.pop(path, None)
        if fh:
            fh.close()
        try:
            self.os.remove(path)
        except OSError:
            pass

    def _cleanup(self, now):
        """Forget recordings older than keep_sec (called with the lock held)."""
        limit = now - self.keep_sec
        while self.f_times and self.f_times[0] < limit and self.f_index[0][0] != self._v_path:
            path = self.f_index[0][0]
            k = 0
            while k < len(self.f_index) and self.f_index[k][0] == path:
                k += 1
            if self.f_times[k - 1] >= limit:
                break
            del self.f_times[:k]
            del self.f_index[:k]
            self._drop(path)
        while (self.base is not None and len(self.a_chunks) > 1
               and self.base + self.a_chunks[0][1] / RATE < limit):
            self._drop(self.a_chunks.pop(0)[2])

    def close(self):
        import shutil
        with self.lock:
            for f in (self._a_file, self._v_file):
                if f:
                    f.close()
            for fh in self._readers.values():
                fh.close()
            self._readers.clear()
        shutil.rmtree(self.dir, ignore_errors=True)

    @staticmethod
    def clean_stale():
        """Remove recordings left behind by a crash."""
        import glob
        import os
        import shutil
        import tempfile
        for d in glob.glob(os.path.join(tempfile.gettempdir(), "japo_video_*")):
            shutil.rmtree(d, ignore_errors=True)


class MemoryStore:
    """Same interface as MediaStore, but RAM only and just long enough for the
    delay (rewinding off: no temporary files)."""

    def __init__(self, keep_sec=70):
        self.keep_sec = keep_sec
        self.lock = threading.Lock()
        self.base = None
        self.n = 0
        self.a_chunks = []               # [n0, array int16 (frames, 2)]
        self.frames = []                 # (t, jpeg)

    def add_audio(self, t_end, pcm16):
        frames = len(pcm16)
        with self.lock:
            if self.base is None:
                self.base = t_end - frames / RATE
            gap = (t_end - frames / RATE) - (self.base + self.n / RATE)
            if gap > 0.08:
                pad = int(gap * RATE)
                self.a_chunks.append([self.n, np.zeros((pad, CHANNELS), np.int16)])
                self.n += pad
            elif gap < -0.5:
                self.base += gap
            self.a_chunks.append([self.n, pcm16])
            self.n += frames
            keep = self.n - int(self.keep_sec * RATE)
            while self.a_chunks and self.a_chunks[0][0] + len(self.a_chunks[0][1]) < keep:
                self.a_chunks.pop(0)

    def read_audio(self, t, frames):
        out = np.zeros((frames, CHANNELS), np.float32)
        with self.lock:
            if self.base is None:
                return out
            start = int(round((t - self.base) * RATE))
            end = start + frames
            for n0, arr in self.a_chunks:
                n1 = n0 + len(arr)
                if n1 <= start or n0 >= end:
                    continue
                a, b = max(start, n0), min(end, n1)
                out[a - start:b - start] = arr[a - n0:b - n0] / 32767.0
        return out

    def add_frame(self, t, jpeg):
        with self.lock:
            self.frames.append((t, jpeg))
            while self.frames and self.frames[0][0] < t - self.keep_sec:
                self.frames.pop(0)

    def frame_at(self, t):
        import bisect
        with self.lock:
            i = bisect.bisect_right(self.frames, t, key=lambda f: f[0]) - 1
            return self.frames[i] if i >= 0 else None

    def oldest(self):
        with self.lock:
            return self.frames[0][0] if self.frames else None

    def close(self):
        with self.lock:
            self.frames.clear()
            self.a_chunks.clear()


# ---------------------------------------------------------------- screen
class ScreenGrabber(threading.Thread):
    """Grab a screen region `fps` times a second into the MediaStore as JPEG."""

    def __init__(self, region, store, fps=25, max_width=1280):
        super().__init__(daemon=True, name="screen")
        self.region = region             # (left, top, width, height) in physical pixels
        self.store = store
        self.fps = fps
        self.max_width = max_width
        self.running = True
        self.error = None

    def run(self):
        try:
            import mss
            from PIL import Image
            left, top, w, h = self.region
            mon = {"left": left, "top": top, "width": w, "height": h}
            scale = min(1.0, self.max_width / max(1, w))
            size = (max(2, int(w * scale)), max(2, int(h * scale)))
            step = 1.0 / self.fps
            with mss.mss() as sct:
                nxt = time.time()
                while self.running:
                    t = time.time()
                    shot = sct.grab(mon)
                    img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
                    if scale < 1.0:
                        img = img.resize(size, Image.BILINEAR)
                    buf = io.BytesIO()
                    img.save(buf, "JPEG", quality=78)
                    self.store.add_frame(t, buf.getvalue())
                    nxt += step
                    pause = nxt - time.time()
                    if pause > 0:
                        time.sleep(pause)
                    else:
                        nxt = time.time()            # too slow: don't try to catch up
        except Exception as e:
            self.error = f"Screen capture failed: {e}"
            print(self.error, flush=True)

    def stop(self):
        self.running = False


# ---------------------------------------------------------------- subtitle rendering
_FONT_FILES = {
    "latin": ["segoeuib.ttf", "seguisb.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"],
    "ja": ["YuGothB.ttc", "meiryob.ttc", "meiryo.ttc", "msgothic.ttc", "NotoSansCJK-Bold.ttc"],
    "ko": ["malgunbd.ttf", "malgun.ttf", "NotoSansCJK-Bold.ttc"],
    "zh": ["msyhbd.ttc", "msyh.ttc", "simhei.ttf", "NotoSansCJK-Bold.ttc"],
    "th": ["LeelaUIb.ttf", "leelawdb.ttf", "tahomabd.ttf"],
}
_font_cache = {}


def _font_path(lang):
    import os
    group = lang.split("-")[0] if lang.split("-")[0] in ("ja", "ko", "zh", "th") else "latin"
    if group in _font_cache:
        return _font_cache[group]
    dirs = [os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"),
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "Windows", "Fonts"),
            "/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/opentype/noto"]
    found = None
    for name in _FONT_FILES[group] + _FONT_FILES["latin"]:
        for d in dirs:
            p = os.path.join(d, name)
            if os.path.isfile(p):
                found = p
                break
        if found:
            break
    _font_cache[group] = found
    return found


def _font(lang, size):
    from PIL import ImageFont
    path = _font_path(lang)
    try:
        return ImageFont.truetype(path, size) if path else ImageFont.load_default(size)
    except Exception:
        return ImageFont.load_default()


def _wrap(text, font, width):
    """Greedy wrap; languages without spaces (ja/zh) wrap per character."""
    tokens = text.split(" ") if " " in text.strip() else list(text)
    sep = " " if " " in text.strip() else ""
    lines, cur = [], ""
    for tok in tokens:
        cand = (cur + sep + tok) if cur else tok
        if font.getlength(cand) <= width or not cur:
            cur = cand
        else:
            lines.append(cur)
            cur = tok
    if cur:
        lines.append(cur)
    return lines


def render_subtitle(text, small, box_w, box_h, max_size, target="vi", source="ja"):
    """Largest text that fits box_w x box_h (translation, plus an optional smaller
    original line above), on a rounded semi-transparent background."""
    from PIL import Image, ImageDraw
    pad = max(6, box_h // 14)
    lo, hi, best = 10, max(11, max_size), None
    while lo <= hi:
        size = (lo + hi) // 2
        f_main = _font(target, size)
        f_small = _font(source, max(9, int(size * 0.6)))
        main = _wrap(text, f_main, box_w - 2 * pad)
        sm = _wrap(small, f_small, box_w - 2 * pad) if small else []
        lh, sh = int(size * 1.25), int(size * 0.6 * 1.3)
        h = len(main) * lh + len(sm) * sh + (pad // 2 if sm else 0) + 2 * pad
        w = max([f_main.getlength(l) for l in main] + [f_small.getlength(l) for l in sm] + [1]) + 2 * pad
        if h <= box_h and w <= box_w:
            best = (size, f_main, f_small, main, sm, lh, sh, int(w), h)
            lo = size + 1
        else:
            hi = size - 1
    if best is None:                                       # box too small: smallest size, clipped
        size = 10
        f_main, f_small = _font(target, size), _font(source, 9)
        main, sm = _wrap(text, f_main, box_w - 2 * pad), []
        lh, sh = int(size * 1.25), 12
        best = (size, f_main, f_small, main, sm, lh, sh, box_w, min(box_h, len(main) * lh + 2 * pad))
    size, f_main, f_small, main, sm, lh, sh, _, text_h = best
    # The panel fills the whole box, so it can cover subtitles burned into the video
    w, h = box_w, box_h
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, w - 1, h - 1), radius=max(6, min(w, h) // 10), fill=(0, 0, 0, 130))
    y = max(pad, (h - text_h) // 2 + pad)
    stroke = max(1, size // 18)
    for line in sm:
        d.text(((w - f_small.getlength(line)) / 2, y), line, font=f_small, fill=(205, 210, 218, 255))
        y += sh
    if sm:
        y += pad // 2
    for line in main:
        d.text(((w - f_main.getlength(line)) / 2, y), line, font=f_main, fill=(255, 255, 255, 255),
               stroke_width=stroke, stroke_fill=(0, 0, 0, 255))
        y += lh
    return img


def paste_frosted(canvas, patch, pos, radius=10):
    """Paste the subtitle on a blurred copy of the video behind it (frosted glass),
    so text burned into the video doesn't show through."""
    from PIL import Image, ImageFilter
    x, y = pos
    box = (max(0, x), max(0, y), min(canvas.width, x + patch.width), min(canvas.height, y + patch.height))
    if box[2] <= box[0] or box[3] <= box[1]:
        return
    region = canvas.crop(box)
    # Blur at 1/4 size and scale back up: same look as a 10 px blur, much faster
    small = region.resize((max(1, region.width // 4), max(1, region.height // 4)), Image.BILINEAR)
    small = small.filter(ImageFilter.GaussianBlur(radius / 4))
    blurred = small.resize(region.size, Image.BILINEAR)
    p = patch.crop((box[0] - x, box[1] - y, box[2] - x, box[3] - y))
    mask = p.getchannel("A").point(lambda a: 255 if a else 0)
    region.paste(blurred, (0, 0), mask)
    region.paste(p, (0, 0), p)
    canvas.paste(region, box[:2])


# ---------------------------------------------------------------- engine source
class _EngineSource:
    """Looks like LoopbackCapture to SubtitleEngine (rate, channels, device, stop)."""

    def __init__(self, name, player):
        self.rate, self.channels = RATE, CHANNELS
        self.device = {"name": f"{name} (app audio)"}
        self._player = player

    def start(self):
        pass

    def stop(self):
        pass                             # the player owns the real capture


# ---------------------------------------------------------------- player
class VideoPlayer:
    def __init__(self, engine, app_name, region, delay, restore_file, fps=25, keep_min=15):
        self.engine = engine
        self.app_name = app_name
        self.region = region
        self.delay = float(delay)
        self.restore_file = restore_file
        self.fps = fps
        self.keep_min = keep_min
        self.store = None                # MediaStore, created in start()
        self.grabber = None
        self.t_start = None              # wall time of the first captured audio
        self.play_t = None               # media time being played (wall clock of capture)
        self.paused = False
        self.waiting = False             # holding for a translation
        self._wait_since = None
        self._skip_upto = 0.0            # pending lines starting before this are not waited for
        self.gain = 1.0 / APP_VOLUME
        self.volume = 1.0                # playback volume set by the UI slider (0..2)
        self.subs = []                   # dicts: start, end, text, orig
        self.subs_lock = threading.Lock()
        self.error = None
        self.ready = threading.Event()
        self._stop = threading.Event()
        self._img_q = queue.Queue(maxsize=1)
        self.view_size = (960, 540)
        self.style = {}
        self._patch_cache = {}
        self._stream = None
        self._pa = None

    # -------------------------------------------------- lifecycle
    def start(self):
        MediaStore.clean_stale()
        if self.keep_min > 0:
            self.store = MediaStore(keep_sec=self.keep_min * 60)
        else:                                            # rewinding off: RAM only, no temp files
            self.store = MemoryStore(keep_sec=self.delay + 60)
        threading.Thread(target=self._audio_worker, daemon=True, name="video-com").start()
        self.grabber = ScreenGrabber(self.region, self.store, self.fps)
        self.grabber.start()
        threading.Thread(target=self._decoder, daemon=True, name="video-decode").start()
        try:
            import pyaudiowpatch as pyaudio
            self._pa = pyaudio.PyAudio()
            self._stream = self._pa.open(format=pyaudio.paFloat32, channels=CHANNELS, rate=RATE,
                                         output=True, frames_per_buffer=BLOCK,
                                         stream_callback=self._out_cb)
            self._stream.start_stream()
        except Exception as e:
            self.error = f"Audio output failed: {e}"

    def _audio_worker(self):
        """COM thread: app capture + keeping the app turned down."""
        cap = vol = None
        try:
            import japo_wasapi as wa
            wa.com_init()
            pid = wa.find_root_pid(self.app_name)
            if not pid:
                raise RuntimeError(f"{self.app_name} is not running")
            vol = wa.AppVolume(self.app_name, APP_VOLUME, self.restore_file)
            vol.apply()
            self.gain = vol.gain()
            cap = wa.ProcessAudioCapture(pid, self._on_audio)
            cap.start()
            self.ready.set()
            while not self._stop.wait(1.0):
                try:
                    vol.apply()                    # new tabs open new audio sessions
                    self.gain = vol.gain()
                except Exception as e:
                    print("Volume control:", e, flush=True)
        except Exception as e:
            self.error = f"App audio failed: {e}"
            print(self.error, flush=True)
            self.ready.set()
        finally:
            if cap:
                cap.stop()
            if vol:
                vol.restore()

    def _on_audio(self, t_end, pcm):
        if self.t_start is None:
            self.t_start = t_end - len(pcm) / RATE
        # Boost back to the original loudness (the app itself runs at 5 %)
        pcm16 = (np.clip(pcm * self.gain, -1.0, 1.0) * 32767).astype(np.int16)
        self.store.add_audio(t_end, pcm16)
        # Feed the subtitle engine (same format its loopback capture produces)
        self.engine.audio_q.put((pcm16.tobytes(), RATE, CHANNELS))

    def engine_source(self):
        return _EngineSource(self.app_name, self)

    def stop(self):
        self._stop.set()
        if self.grabber:
            self.grabber.stop()
        if self.store:
            threading.Timer(0.6, self.store.close).start()      # after the threads have stopped
        try:
            if self._stream:
                self._stream.stop_stream()
                self._stream.close()
            if self._pa:
                self._pa.terminate()
        except Exception:
            pass

    # -------------------------------------------------- playback clock (audio callback)
    def _out_cb(self, in_data, frame_count, time_info, status):
        import pyaudiowpatch as pyaudio
        out = self._next_audio(frame_count, time.time())
        return (out.tobytes(), pyaudio.paContinue)

    def _next_audio(self, n, now):
        silence = np.zeros((n, CHANNELS), np.float32)
        if self.t_start is None or self.paused:
            return silence
        if self.play_t is None:
            if now - self.delay < self.t_start:
                return silence                       # still filling the buffer
            self.play_t = self.t_start
        dur = n / RATE
        limit = now - self.delay + 0.25              # never closer to live than `delay`
        gate = self.engine.earliest_pending()
        gated = gate is not None and gate > self._skip_upto and gate < limit
        if gated:
            limit = gate
        if self.play_t + dur <= limit:
            out = self.store.read_audio(self.play_t, n) * self.volume
            self.play_t += dur
            self.waiting = False
            self._wait_since = None
            return np.clip(out, -1.0, 1.0).astype(np.float32)
        if gated:
            self.waiting = True
            if self._wait_since is None:
                self._wait_since = now
            elif now - self._wait_since > MAX_WAIT:      # give up on this line
                self._skip_upto = gate + 0.5
                self._wait_since = None
        return silence

    def current_delay(self):
        return None if self.play_t is None else time.time() - self.play_t

    def buffering_left(self):
        if self.t_start is None:
            return self.delay
        return max(0.0, self.t_start + self.delay - time.time())

    # -------------------------------------------------- video frames
    def set_view_size(self, w, h):
        self.view_size = (max(2, w), max(2, h))

    def set_style(self, **kw):
        """Subtitle look, set by the UI: box (x0, y0, x1, y1 as 0..1 of the view),
        max_size (px), show_orig, use_roman, target, source."""
        self.style.update(kw)

    def set_region(self, region):
        """Capture a new area (the recording keeps the old one up to now)."""
        if self.grabber:
            self.grabber.stop()
        self.region = region
        self.grabber = ScreenGrabber(region, self.store, self.fps)
        self.grabber.start()

    def _frame_at(self, t):
        try:
            return self.store.frame_at(t) if self.store else None
        except (OSError, ValueError):
            return None                                  # closing

    # -------------------------------------------------- seeking
    def live_t(self):
        """Latest position playback may reach (`delay` behind real time)."""
        return time.time() - self.delay

    def time_range(self):
        lo = (self.store.oldest() if self.store else None) or self.t_start or self.live_t()
        return lo, max(lo, self.live_t())

    def seek(self, t):
        if self.play_t is None:
            return
        lo, hi = self.time_range()
        self.play_t = min(max(t, lo), hi)
        self.waiting = False
        self._wait_since = None

    def can_rewind(self):
        return isinstance(self.store, MediaStore)

    def is_live(self):
        return self.play_t is not None and self.play_t >= self.live_t() - 2.0

    def ready_to_play(self):
        return self.play_t is not None and not self.error

    def _decoder(self):
        """Compose (video frame + subtitle) at view size off the UI thread."""
        from PIL import Image
        scaled = {"t": None, "size": None, "img": None}
        last_key = None
        while not self._stop.is_set():
            t = self.play_t
            fr = self._frame_at(t) if t is not None else None
            sub = self.current_sub()
            vw, vh = self.view_size
            style_key = tuple(sorted((k, str(v)) for k, v in self.style.items()))
            key = (fr[0] if fr else None, id(sub) if sub else None, (vw, vh), style_key)
            if fr is not None and key != last_key:
                try:
                    ft, data = fr
                    if scaled["t"] != ft or scaled["size"] != (vw, vh):
                        img = Image.open(io.BytesIO(data))
                        k = min(vw / img.width, vh / img.height)
                        size = (max(2, int(img.width * k)), max(2, int(img.height * k)))
                        img.draft("RGB", size)
                        scaled.update(t=ft, size=(vw, vh), img=img.convert("RGB").resize(size, Image.BILINEAR))
                    frame = scaled["img"]
                    canvas = Image.new("RGB", (vw, vh))
                    canvas.paste(frame, ((vw - frame.width) // 2, (vh - frame.height) // 2))
                    if sub:
                        patch, pos = self._sub_patch(sub, vw, vh)
                        if patch is not None:
                            paste_frosted(canvas, patch, pos)
                    try:
                        self._img_q.get_nowait()
                    except queue.Empty:
                        pass
                    self._img_q.put_nowait(canvas)
                    last_key = key
                except Exception as e:
                    print("Frame compose:", e, flush=True)
            time.sleep(0.008)

    def _sub_patch(self, sub, vw, vh):
        """Rendered subtitle (RGBA, semi-transparent background) that fits the box."""
        st = self.style
        box = st.get("box") or (0.12, 0.76, 0.88, 0.96)
        bx0, by0 = int(box[0] * vw), int(box[1] * vh)
        bw, bh = max(40, int((box[2] - box[0]) * vw)), max(24, int((box[3] - box[1]) * vh))
        small = ""
        if st.get("show_orig"):
            small = sub["roman"] if (st.get("use_roman") and sub.get("roman")) else sub.get("orig", "")
        key = (sub["text"], small, bw, bh, st.get("max_size"), st.get("target"), st.get("source"))
        cache = self._patch_cache
        if key not in cache:
            if len(cache) > 30:
                cache.clear()
            cache[key] = render_subtitle(sub["text"], small, bw, bh, int(st.get("max_size", 40)),
                                         st.get("target", "vi"), st.get("source", "ja"))
        patch = cache[key]
        if patch is None:
            return None, None
        x, y = bx0, by0
        return patch, (x, y)

    def take_image(self):
        try:
            return self._img_q.get_nowait()
        except queue.Empty:
            return None

    # -------------------------------------------------- subtitles
    def add_line(self, start, end, text, orig="", roman=""):
        with self.subs_lock:
            self.subs.append({"start": start, "end": end, "text": text, "orig": orig, "roman": roman})
            self.subs.sort(key=lambda s: s["start"])
            if len(self.subs) > 400:
                del self.subs[:100]

    def current_sub(self):
        t = self.play_t
        if t is None:
            return None
        with self.subs_lock:
            cur = None
            for i, s in enumerate(self.subs):
                if s["start"] - 0.15 > t:
                    break
                nxt = self.subs[i + 1]["start"] if i + 1 < len(self.subs) else None
                end = max(s["end"], s["start"] + 1.5) + 0.4      # short lines stay readable
                if nxt is not None:
                    end = min(end, max(nxt - 0.05, s["end"]))
                if t <= end:
                    cur = s
        return cur
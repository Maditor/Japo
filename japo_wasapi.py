"""
Windows audio helpers for Japo's video mode (Windows 10 2004+ / Windows 11).

- ProcessAudioCapture: capture the audio of one app (and its child processes)
  through WASAPI "process loopback", as 48 kHz stereo float32.
- AppVolume: turn an app down in the Volume Mixer and restore it afterwards.

Import this module only from a worker thread: COM is initialised as
multithreaded here, which the process-loopback API requires and which must
not happen on Tk's main thread.
"""
import sys
sys.coinit_flags = 0                      # COINIT_MULTITHREADED for the importing thread

import ctypes
import json
import os
import threading
import time
from ctypes import POINTER, Structure, byref, c_byte, c_int, c_long, c_uint16, c_uint32, c_uint64, \
    c_ulong, c_ushort, c_void_p, c_wchar_p, HRESULT
from ctypes.wintypes import DWORD

import numpy as np
import comtypes
from comtypes import COMMETHOD, GUID, IUnknown

RATE, CHANNELS = 48000, 2


def com_init():
    """Call at the start of every thread that uses this module."""
    try:
        comtypes.CoInitializeEx(0)
    except OSError:
        pass                                # already initialised in this thread


# ---------------------------------------------------------------- COM interfaces
class IAgileObject(IUnknown):
    _iid_ = GUID("{94ea2b94-e9cc-49e0-c0ff-ee64ca8f5b90}")
    _methods_ = []


class IAudioClient(IUnknown):
    _iid_ = GUID("{1CB9AD4C-DBFA-4c32-B178-C2F568A703B2}")


class IAudioCaptureClient(IUnknown):
    _iid_ = GUID("{C8ADBD64-E71E-48a0-A4DE-185C395CD317}")


class IActivateAudioInterfaceAsyncOperation(IUnknown):
    _iid_ = GUID("{72A22D78-CDE4-431D-B8CC-843A71199B6D}")


class IActivateAudioInterfaceCompletionHandler(IUnknown):
    _iid_ = GUID("{41D949AB-9862-444A-80F6-C261334DA5EB}")


class WAVEFORMATEX(Structure):
    _pack_ = 1
    _fields_ = [("wFormatTag", c_uint16), ("nChannels", c_uint16), ("nSamplesPerSec", c_uint32),
                ("nAvgBytesPerSec", c_uint32), ("nBlockAlign", c_uint16), ("wBitsPerSample", c_uint16),
                ("cbSize", c_uint16)]


IAudioClient._methods_ = [
    COMMETHOD([], HRESULT, "Initialize",
              (["in"], c_int, "ShareMode"), (["in"], DWORD, "StreamFlags"),
              (["in"], ctypes.c_int64, "hnsBufferDuration"), (["in"], ctypes.c_int64, "hnsPeriodicity"),
              (["in"], POINTER(WAVEFORMATEX), "pFormat"), (["in"], POINTER(GUID), "AudioSessionGuid")),
    COMMETHOD([], HRESULT, "GetBufferSize", (["out"], POINTER(c_uint32), "frames")),
    COMMETHOD([], HRESULT, "GetStreamLatency", (["out"], POINTER(ctypes.c_int64), "latency")),
    COMMETHOD([], HRESULT, "GetCurrentPadding", (["out"], POINTER(c_uint32), "padding")),
    COMMETHOD([], HRESULT, "IsFormatSupported", (["in"], c_int, "mode"), (["in"], c_void_p, "fmt"),
              (["in"], c_void_p, "closest")),
    COMMETHOD([], HRESULT, "GetMixFormat", (["in"], c_void_p, "fmt")),
    COMMETHOD([], HRESULT, "GetDevicePeriod", (["in"], c_void_p, "a"), (["in"], c_void_p, "b")),
    COMMETHOD([], HRESULT, "Start"),
    COMMETHOD([], HRESULT, "Stop"),
    COMMETHOD([], HRESULT, "Reset"),
    COMMETHOD([], HRESULT, "SetEventHandle", (["in"], c_void_p, "handle")),
    COMMETHOD([], HRESULT, "GetService", (["in"], POINTER(GUID), "riid"),
              (["out"], POINTER(POINTER(IUnknown)), "ppv")),
]
IAudioCaptureClient._methods_ = [
    COMMETHOD([], HRESULT, "GetBuffer", (["out"], POINTER(POINTER(c_byte)), "data"),
              (["out"], POINTER(c_uint32), "frames"), (["out"], POINTER(c_uint32), "flags"),
              (["out"], POINTER(c_uint64), "devpos"), (["out"], POINTER(c_uint64), "qpcpos")),
    COMMETHOD([], HRESULT, "ReleaseBuffer", (["in"], c_uint32, "frames")),
    COMMETHOD([], HRESULT, "GetNextPacketSize", (["out"], POINTER(c_uint32), "frames")),
]
IActivateAudioInterfaceAsyncOperation._methods_ = [
    COMMETHOD([], HRESULT, "GetActivateResult", (["out"], POINTER(c_long), "hr"),
              (["out"], POINTER(POINTER(IUnknown)), "iface")),
]
IActivateAudioInterfaceCompletionHandler._methods_ = [
    COMMETHOD([], HRESULT, "ActivateCompleted",
              (["in"], POINTER(IActivateAudioInterfaceAsyncOperation), "op")),
]


class LOOPBACK_PARAMS(Structure):
    _fields_ = [("TargetProcessId", DWORD), ("ProcessLoopbackMode", c_int)]


class ACTIVATION_PARAMS(Structure):
    _fields_ = [("ActivationType", c_int), ("ProcessLoopbackParams", LOOPBACK_PARAMS)]


class BLOB(Structure):
    _fields_ = [("cbSize", c_ulong), ("pBlobData", c_void_p)]


class PROPVARIANT(Structure):
    _fields_ = [("vt", c_ushort), ("r1", c_ushort), ("r2", c_ushort), ("r3", c_ushort), ("blob", BLOB)]


class _CompletionHandler(comtypes.COMObject):
    _com_interfaces_ = [IActivateAudioInterfaceCompletionHandler, IAgileObject]

    def __init__(self):
        super().__init__()
        self.done = threading.Event()
        self.client = None
        self.error = None

    def IActivateAudioInterfaceCompletionHandler_ActivateCompleted(self, this, op):
        try:
            hr, unk = op.GetActivateResult()
            if hr < 0:
                self.error = f"activation failed (HRESULT 0x{hr & 0xFFFFFFFF:08X})"
            else:
                self.client = unk.QueryInterface(IAudioClient)
        except Exception as e:
            self.error = repr(e)
        self.done.set()
        return 0


_STREAM_FLAGS = 0x00020000 | 0x00040000 | 0x80000000 | 0x08000000   # loopback|event|autoconvert|SRC quality


# ---------------------------------------------------------------- processes
def find_root_pid(name):
    """PID of the top process named `name` (e.g. brave.exe). Its whole process
    tree is captured, so every tab's audio is included."""
    import psutil
    name = name.lower()
    best = None
    for p in psutil.process_iter(["name", "pid"]):
        if (p.info["name"] or "").lower() != name:
            continue
        try:
            parent = p.parent()
            if parent is None or (parent.name() or "").lower() != name:
                return p.info["pid"]
        except Exception:
            pass
        best = best or p.info["pid"]
    return best


def list_audio_apps():
    """Names of apps that currently have an audio session, e.g. ['brave.exe', 'Spotify.exe']."""
    from pycaw.pycaw import AudioUtilities
    names = []
    for s in AudioUtilities.GetAllSessions():
        try:
            if s.Process:
                n = s.Process.name()
                if n not in names:
                    names.append(n)
        except Exception:
            pass
    return names


# ---------------------------------------------------------------- capture
class ProcessAudioCapture:
    """Capture one app's audio. on_audio(t_end, pcm) gets float32 arrays (frames, 2)
    together with the wall-clock time (time.time()) at which the packet ended."""

    def __init__(self, pid, on_audio):
        self.pid = pid
        self.on_audio = on_audio
        self.running = False
        self.thread = None

    def _activate(self):
        params = ACTIVATION_PARAMS(1, LOOPBACK_PARAMS(self.pid, 0))          # 0 = include process tree
        pv = PROPVARIANT(65, 0, 0, 0, BLOB(ctypes.sizeof(params), ctypes.cast(byref(params), c_void_p)))
        handler = _CompletionHandler()
        op = POINTER(IActivateAudioInterfaceAsyncOperation)()
        fn = ctypes.OleDLL("Mmdevapi.dll").ActivateAudioInterfaceAsync
        fn.argtypes = [c_wchar_p, POINTER(GUID), POINTER(PROPVARIANT),
                       POINTER(IActivateAudioInterfaceCompletionHandler),
                       POINTER(POINTER(IActivateAudioInterfaceAsyncOperation))]
        fn("VAD\\Process_Loopback", byref(IAudioClient._iid_), byref(pv),
           handler.QueryInterface(IActivateAudioInterfaceCompletionHandler), byref(op))
        if not handler.done.wait(5):
            raise RuntimeError("Windows did not answer the app-audio request (needs Windows 10 2004+)")
        if handler.error:
            raise RuntimeError(handler.error)
        return handler.client

    def start(self):
        com_init()
        self.client = self._activate()
        fmt = WAVEFORMATEX(3, CHANNELS, RATE, RATE * CHANNELS * 4, CHANNELS * 4, 32, 0)   # float32
        self.client.Initialize(0, _STREAM_FLAGS, 2_000_000, 0, byref(fmt), None)
        self.event = ctypes.windll.kernel32.CreateEventW(None, False, False, None)
        self.client.SetEventHandle(self.event)
        self.capture = self.client.GetService(byref(IAudioCaptureClient._iid_)).QueryInterface(
            IAudioCaptureClient)
        self.client.Start()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True, name="app-audio")
        self.thread.start()

    def _loop(self):
        com_init()
        wait = ctypes.windll.kernel32.WaitForSingleObject
        while self.running:
            wait(self.event, 100)
            try:
                while self.running:
                    n = self.capture.GetNextPacketSize()
                    if not n:
                        break
                    data, frames, flags, _, _ = self.capture.GetBuffer()
                    if flags & 0x2:                                          # silent packet
                        pcm = np.zeros((frames, CHANNELS), np.float32)
                    else:
                        raw = ctypes.string_at(data, frames * CHANNELS * 4)
                        pcm = np.frombuffer(raw, np.float32).reshape(-1, CHANNELS).copy()
                    self.capture.ReleaseBuffer(frames)
                    self.on_audio(time.time(), pcm)
            except Exception as e:
                print("App audio capture error:", e, flush=True)
                time.sleep(0.2)

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(1)
        try:
            self.client.Stop()
        except Exception:
            pass


# ---------------------------------------------------------------- volume
class AppVolume:
    """Keep every audio session of an app at `level` and restore the original
    volumes afterwards. Originals are also written to a file, so a crash never
    leaves the browser stuck at 5 %."""

    def __init__(self, name, level, restore_file):
        self.name = name.lower()
        self.level = level
        self.file = restore_file
        self.original = {}               # session instance id -> original level
        try:
            with open(self.file, encoding="utf-8") as f:
                saved = json.load(f)
            if saved.get("app") == self.name:
                self.fallback = float(saved.get("level", 1.0))
            else:
                self.fallback = None
        except Exception:
            self.fallback = None

    def _sessions(self):
        from pycaw.pycaw import AudioUtilities
        out = []
        for s in AudioUtilities.GetAllSessions():
            try:
                if s.Process and s.Process.name().lower() == self.name:
                    out.append(s)
            except Exception:
                pass
        return out

    def apply(self):
        """Turn new sessions down (call every second or so)."""
        for s in self._sessions():
            key = s.InstanceIdentifier
            v = s.SimpleAudioVolume
            if key not in self.original:
                cur = v.GetMasterVolume()
                if abs(cur - self.level) < 0.005 and self.fallback:      # left low by a crash
                    cur = self.fallback
                self.original[key] = cur
                self._save(cur)
            if abs(v.GetMasterVolume() - self.level) > 0.002:
                v.SetMasterVolume(self.level, None)
            if v.GetMute():
                v.SetMute(0, None)

    def gain(self):
        """Factor that brings the captured audio back to the original loudness."""
        orig = max(self.original.values()) if self.original else (self.fallback or 1.0)
        return max(1.0, orig / self.level)

    def restore(self):
        for s in self._sessions():
            orig = self.original.get(s.InstanceIdentifier, self.fallback)
            if orig is not None:
                try:
                    s.SimpleAudioVolume.SetMasterVolume(float(orig), None)
                except Exception:
                    pass
        try:
            os.remove(self.file)
        except OSError:
            pass

    def _save(self, level):
        try:
            with open(self.file, "w", encoding="utf-8") as f:
                json.dump({"app": self.name, "level": level}, f)
        except OSError:
            pass

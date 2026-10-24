"""Low-latency audio capture and playback using sounddevice (PortAudio)."""
from __future__ import annotations

import queue
import threading
from typing import Callable, Optional

import numpy as np

try:
    import sounddevice as sd
except Exception:  # pragma: no cover - optional dependency
    sd = None


class AudioPlayer:
    """Streaming playback queue. Chunks are played in order; clear() drops them."""

    def __init__(self, sample_rate: int = 24000, block_size: int = 1024, gain: float = 1.5, device=None):
        self.sample_rate = int(sample_rate)
        self.block_size = int(block_size)
        self.gain = float(gain)
        self.device = device
        self._q: "queue.Queue[np.ndarray]" = queue.Queue()
        self._current: Optional[np.ndarray] = None
        self._pos = 0
        self._stream = None
        self._lock = threading.Lock()
        self._playing = False

    def _callback(self, outdata, frames, time_info, status):
        out = np.zeros((frames, 1), dtype=np.float32)
        filled = 0
        while filled < frames:
            if self._current is None or self._pos >= len(self._current):
                try:
                    self._current = self._q.get_nowait()
                except queue.Empty:
                    self._current = None
                    break
                self._pos = 0
            take = min(frames - filled, len(self._current) - self._pos)
            out[filled:filled + take, 0] = self._current[self._pos:self._pos + take]
            self._pos += take
            filled += take
        outdata[:] = out
        self._playing = filled > 0

    @property
    def is_playing(self) -> bool:
        with self._lock:
            return self._playing

    def start(self) -> None:
        if sd is None:
            raise RuntimeError("sounddevice is not installed")
        if self._stream is not None:
            return
        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.block_size,
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
            except Exception:
                pass
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        with self._lock:
            self._current = None
            self._pos = 0
            self._playing = False
            self._playing = False
            while not self._q.empty():
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    break

    def play(self, audio: np.ndarray) -> None:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if not audio.size:
            return
        if self.gain != 1.0:
            audio = np.tanh(audio * self.gain).astype(np.float32)
        with self._lock:
            self._q.put(audio)

    def clear(self) -> None:
        with self._lock:
            while not self._q.empty():
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    break
            self._current = None
            self._pos = 0
            self._playing = False


class MicCapture:
    """Continuous 16 kHz mono capture; callback receives float32 numpy chunks."""

    def __init__(self, sample_rate: int = 16000, block_size: int = 512,
                 callback: Optional[Callable[[np.ndarray], None]] = None, device=None):
        self.sample_rate = int(sample_rate)
        self.block_size = int(block_size)
        self.callback = callback
        self.device = device
        self._stream = None

    def _cb(self, indata, frames, time_info, status):
        if self.callback is None:
            return
        try:
            self.callback(np.asarray(indata[:, 0], dtype=np.float32).copy())
        except Exception:
            pass

    def start(self) -> None:
        if sd is None:
            raise RuntimeError("sounddevice is not installed")
        if self._stream is not None:
            return
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.block_size,
            device=self.device,
            callback=self._cb,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
            except Exception:
                pass
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        with self._lock:
            self._current = None
            self._pos = 0
            self._playing = False
            self._playing = False
            while not self._q.empty():
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    break

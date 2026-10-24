"""Cancellable TTS worker with sentence/segment streaming playback."""
from __future__ import annotations

import queue
import threading
from typing import Optional

import numpy as np

from pipeline.tts.qwen3 import EmotionTTS


def _resample(audio: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if sr == target_sr or audio.size == 0:
        return audio
    try:
        import librosa

        return librosa.resample(audio, orig_sr=sr, target_sr=target_sr).astype("float32")
    except Exception:
        return audio


class TTSWorker:
    def __init__(self, tts: EmotionTTS, player, target_sample_rate: int = 24000):
        self.tts = tts
        self.player = player
        self.target_sample_rate = int(target_sample_rate)
        self._q: "queue.Queue" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._q.put(None)
        if self._thread is not None:
            self._thread.join(timeout=5)

    def speak(self, tagged_reply: str, epoch: Optional[int] = None) -> None:
        if epoch is None:
            epoch = self.tts.guard.epoch
        self._q.put((tagged_reply, epoch))

    def cancel(self) -> int:
        new_epoch = self.tts.cancel()
        # Drain queue.
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        try:
            self.player.clear()
        except Exception:
            pass
        return new_epoch

    def _run(self) -> None:
        while self._running:
            item = self._q.get()
            if item is None:
                break
            text, epoch = item
            if not self.tts.guard.is_current(epoch):
                continue
            try:
                for audio, sr in self.tts.render_streaming(text):
                    if not self.tts.guard.is_current(epoch):
                        break
                    if not len(audio):
                        continue
                    self.player.play(_resample(audio, int(sr), self.target_sample_rate))
            except Exception as exc:
                print(f"[tts] synthesis failed: {exc}")

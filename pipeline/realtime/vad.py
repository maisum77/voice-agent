"""Voice activity detection helpers."""
from __future__ import annotations

import numpy as np


class EnergyVAD:
    def __init__(self, threshold: float = 0.015):
        self.threshold = float(threshold)

    def reset(self) -> None:
        pass

    def is_speech(self, frame: np.ndarray, sr: int = 16000) -> bool:
        frame = np.asarray(frame, dtype=np.float32).reshape(-1)
        if frame.size == 0:
            return False
        rms = float(np.sqrt(np.mean(frame * frame)))
        return rms >= self.threshold


class SileroVAD:
    def __init__(self, threshold: float = 0.5):
        import torch

        self._torch = torch
        model, _ = torch.hub.load(
            "snakers4/silero-vad", "silero_vad", trust_repo=True, verbose=False
        )
        self.model = model
        self.threshold = float(threshold)
        self.model.eval()

    def reset(self) -> None:
        try:
            self.model.reset_states()
        except Exception:
            pass

    def is_speech(self, frame: np.ndarray, sr: int = 16000) -> bool:
        frame = np.asarray(frame, dtype=np.float32).reshape(-1)
        if frame.size == 0:
            return False
        # Silero expects 512 samples at 16 kHz.
        if frame.size < 512:
            frame = np.pad(frame, (0, 512 - frame.size))
        elif frame.size > 512:
            frame = frame[-512:]
        with self._torch.no_grad():
            prob = self.model(self._torch.from_numpy(frame), sr).item()
        return bool(prob >= self.threshold)


def make_vad(preferred: str = "auto", threshold: float = 0.5):
    preferred = (preferred or "auto").lower()
    if preferred in ("auto", "silero"):
        try:
            return SileroVAD(threshold=threshold)
        except Exception:
            if preferred == "silero":
                raise
    return EnergyVAD(threshold=0.015)

"""Real-time, interruptible voice loop orchestrator."""
from __future__ import annotations

import queue
import threading
import time
from typing import Callable, Optional

import numpy as np

from pipeline.emotion.guard import apply as apply_emotion_guard
from pipeline.realtime.audio_io import AudioPlayer, MicCapture
from pipeline.realtime.llm_stream import stream_generate
from pipeline.realtime.stt import STTBackend, make_stt
from pipeline.realtime.tts_worker import TTSWorker
from pipeline.realtime.vad import make_vad


class VoiceLoop:
    def __init__(
        self,
        agent,
        system_prompt: str,
        stt: STTBackend,
        vad=None,
        on_user_text: Optional[Callable[[str], None]] = None,
        on_assistant_text: Optional[Callable[[str], None]] = None,
        vad_threshold: float = 0.5,
        silence_ms: int = 700,
        min_utterance_ms: int = 500,
        max_utterance_ms: int = 4000,
        debug: bool = False,
        input_device=None,
        output_device=None,
    ):
        self.agent = agent
        self.system_prompt = system_prompt
        self.stt = stt
        self.vad = vad if vad is not None else make_vad("auto", vad_threshold)
        self.on_user_text = on_user_text or (lambda t: print(f"user> {t}"))
        self.on_assistant_text = on_assistant_text or (lambda t: print(f"assistant> {t}"))

        self.sample_rate = 16000
        self.silence_samples = int(self.sample_rate * silence_ms / 1000)
        self.min_utterance_samples = int(self.sample_rate * min_utterance_ms / 1000)
        self.max_utterance_samples = int(self.sample_rate * max_utterance_ms / 1000)

        self.tts = agent.tts_engine
        if self.tts is None:
            agent.setup_tts()
            self.tts = agent.tts_engine
        if self.tts is None:
            raise RuntimeError("TTS not available - run design_voice.py first")
        # Ensure shared guard.
        self.tts.guard = agent.guard

        self.debug = debug
        self.player = AudioPlayer(sample_rate=24000, device=output_device)
        self.tts_worker = TTSWorker(self.tts, self.player, target_sample_rate=24000)
        self.mic = MicCapture(sample_rate=self.sample_rate, block_size=512, callback=self._on_audio, device=input_device)

        self._utterance = []
        self._speech_active = False
        self._silence_count = 0
        self._speech_run = 0
        self._state = "idle"
        self._utterance_q = queue.Queue()
        self._cancel_event = threading.Event()
        self._running = False
        self._proc_thread = None

    @property
    def state(self) -> str:
        return self._state

    @state.setter
    def state(self, value: str) -> None:
        self._state = value

    def _interrupt(self) -> None:
        self._cancel_event.set()
        self.tts_worker.cancel()
        try:
            self.player.clear()
        except Exception:
            pass
        self.state = "idle"
    def _on_audio(self, chunk: np.ndarray) -> None:
        if not self._running:
            return
        try:
            is_speech = bool(self.vad.is_speech(chunk, self.sample_rate))
        except Exception:
            is_speech = False
        rms = float(np.sqrt(np.mean(chunk * chunk))) if chunk.size else 0.0
        # Reject low-level noise.
        if rms < 0.035:
            is_speech = False
        if is_speech:
            self._speech_run += 1
        else:
            self._speech_run = 0
        # Require a couple of consecutive speech frames to start.
        if not self._speech_active and self._speech_run < 2:
            is_speech = False
        # Energy fallback for very clear speech.
        if not is_speech and rms >= 0.05:
            is_speech = True

        if is_speech:
            if not self._speech_active:
                self._speech_active = True
                self._silence_count = 0
                self._utterance = [chunk.copy()]
                if self.debug:
                    print(f"[vad] speech start (rms={rms:.4f})")
                if self.state in ("speaking", "processing") or self.player.is_playing:
                    self._interrupt()
            else:
                self._utterance.append(chunk.copy())
                self._silence_count = 0
            total = sum(c.size for c in self._utterance)
            if self._speech_active and total >= self.max_utterance_samples:
                self._speech_active = False
                audio = np.concatenate(self._utterance) if self._utterance else np.zeros(0, dtype=np.float32)
                self._utterance = []
                self._speech_run = 0
                try:
                    self.vad.reset()
                except Exception:
                    pass
                if audio.size >= self.min_utterance_samples:
                    self._utterance_q.put(audio)
                    if self.debug:
                        print(f"[vad] utterance queued ({audio.size/16000:.2f}s, max-cap)")
                self._silence_count = 0
        else:
            if self._speech_active:
                self._utterance.append(chunk.copy())
                self._silence_count += len(chunk)
                if self._silence_count >= self.silence_samples:
                    self._speech_active = False
                    audio = np.concatenate(self._utterance) if self._utterance else np.zeros(0, dtype=np.float32)
                    self._utterance = []
                    self._speech_run = 0
                    try:
                        self.vad.reset()
                    except Exception:
                        pass
                    if audio.size >= self.min_utterance_samples:
                        self._utterance_q.put(audio)
                        if self.debug:
                            print(f"[vad] utterance queued ({audio.size/16000:.2f}s)")
                    self._silence_count = 0

    def _process_loop(self) -> None:
        while self._running:
            try:
                audio = self._utterance_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if audio is None:
                break
            self.state = "processing"
            self._cancel_event.clear()
            try:
                text = self.stt.transcribe(audio, self.sample_rate)
            except Exception as exc:
                print(f"[stt] failed: {exc}")
                text = ""
            text = (text or "").strip()
            if not text or len(text) < 2:
                self.state = "idle"
                continue
            if self._cancel_event.is_set():
                self.state = "idle"
                continue
            self.on_user_text(text)

            raw = stream_generate(
                self.agent.model,
                self.agent.tok,
                text,
                self.system_prompt,
                cancel_event=self._cancel_event,
                max_new_tokens=80,
                do_sample=False,
                repetition_penalty=1.05,
            )
            if self._cancel_event.is_set():
                self.state = "idle"
                continue
            raw = (raw or "").strip()
            if not raw:
                self.state = "idle"
                continue

            result = apply_emotion_guard(text, raw)
            tagged = result.text
            self.on_assistant_text(tagged)

            if self._cancel_event.is_set():
                self.state = "idle"
                continue
            self.tts_worker.speak(tagged, epoch=self.agent.guard.epoch)
            self.state = "speaking"

    def run(self) -> None:
        self._running = True
        self.player.start()
        self.tts_worker.start()
        # Warm up TTS so the first reply is not delayed by model load.
        def _warmup():
            try:
                self.tts.render("[emotion:neutral] hi")
            except Exception:
                pass
        threading.Thread(target=_warmup, daemon=True).start()
        self.mic.start()
        self._proc_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._proc_thread.start()
        print("[voice-loop] running - speak now. Ctrl-C to stop.")
        try:
            while self._running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        self.stop()

    def stop(self) -> None:
        self._running = False
        try:
            self.mic.stop()
        except Exception:
            pass
        try:
            self.tts_worker.stop()
        except Exception:
            pass
        try:
            self.player.stop()
        except Exception:
            pass
        if self._proc_thread is not None:
            self._proc_thread.join(timeout=2)
        print("[voice-loop] stopped")


"""Streaming LLM generation helpers (transformers backend)."""
from __future__ import annotations

import threading
from typing import Callable, Optional

import torch
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer


class CancelStoppingCriteria(StoppingCriteria):
    def __init__(self, cancel_event: threading.Event):
        self.cancel_event = cancel_event

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return self.cancel_event.is_set()


def stream_generate(
    model,
    tokenizer,
    user_text: str,
    system_prompt: str,
    cancel_event: Optional[threading.Event] = None,
    max_new_tokens: int = 256,
    on_token: Optional[Callable[[str], None]] = None,
    do_sample: bool = True,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.0,
) -> str:
    cancel_event = cancel_event or threading.Event()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = f"{system_prompt}\n\nUser: {user_text}\nAssistant:"


    inputs = tokenizer(prompt, return_tensors="pt")
    try:
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    except Exception:
        pass

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        streamer=streamer,
        stopping_criteria=StoppingCriteriaList([CancelStoppingCriteria(cancel_event)]),
    )
    if do_sample:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
    else:
        gen_kwargs.update(do_sample=False)
    if repetition_penalty and repetition_penalty != 1.0:
        gen_kwargs["repetition_penalty"] = repetition_penalty

    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs, daemon=True)
    thread.start()

    collected = []
    for token in streamer:
        if cancel_event.is_set():
            break
        collected.append(token)
        if on_token is not None:
            try:
                on_token(token)
            except Exception:
                pass
    thread.join(timeout=10)
    return "".join(collected)

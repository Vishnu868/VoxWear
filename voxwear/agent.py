"""Agentic AI core: context memory + intent understanding + LLM response generation.

Backends:
  "openai:gpt-4o-mini"      (needs OPENAI_API_KEY)
  "ollama:llama3.2:3b"      (local, needs Ollama running on localhost:11434)
"""
import json
import os
import urllib.request
from collections import deque

SYSTEM_PROMPT = """You are VoxWear, a communication assistant worn by a person who may be
speech- or hearing-impaired. Inputs arrive as recognised hand gestures (with an intent),
transcribed speech from people nearby, and a description of objects in view.
Think step by step: (1) what does the user want, (2) what context is available,
(3) what is the most helpful reply. Then answer in ONE or TWO short, polite sentences
suitable to be spoken aloud. Use the environment to give concrete, spatial help.
Never give unsafe medical instructions; for medical alerts, suggest calling for help."""


class VoxWearAgent:
    def __init__(self, backend: str = "openai:gpt-4o-mini", memory_size: int = 10,
                 temperature: float = 0.7, max_tokens: int = 150):
        self.provider, self.model = backend.split(":", 1)
        self.memory = deque(maxlen=memory_size)
        self.environment = "unknown"
        self.temperature, self.max_tokens = temperature, max_tokens

    def update_environment(self, description: str):
        self.environment = description

    def _build_messages(self, user_turn: str):
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        for turn in self.memory:
            msgs.append(turn)
        msgs.append({"role": "user",
                     "content": f"{user_turn}\nEnvironment: {self.environment}"})
        return msgs

    def _call_llm(self, messages) -> str:
        if self.provider == "openai":
            from openai import OpenAI
            r = OpenAI().chat.completions.create(model=self.model, messages=messages,
                                                 temperature=self.temperature, top_p=0.9,
                                                 max_tokens=self.max_tokens)
            return r.choices[0].message.content.strip()
        if self.provider == "ollama":
            payload = json.dumps({"model": self.model, "messages": messages, "stream": False,
                                  "options": {"temperature": self.temperature, "top_p": 0.9,
                                              "num_predict": self.max_tokens}}).encode()
            req = urllib.request.Request(os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat"),
                                         data=payload, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read())["message"]["content"].strip()
        raise ValueError(f"Unknown LLM provider: {self.provider}")

    def respond(self, *, gesture: dict | None = None, speech: str | None = None) -> str:
        if gesture:
            user_turn = (f"[Gesture] '{gesture['gesture']}' (intent: {gesture['intent']}, "
                         f"confidence {gesture['confidence']:.0%})")
        elif speech:
            user_turn = f"[Speech] \"{speech}\""
        else:
            raise ValueError("Need a gesture or speech input")
        reply = self._call_llm(self._build_messages(user_turn))
        self.memory.append({"role": "user", "content": user_turn})
        self.memory.append({"role": "assistant", "content": reply})
        return reply

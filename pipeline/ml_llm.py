"""Minimal Gemini Flash-Lite client (public text only). Key from the GEMINI_API_KEY environment variable.
Every call logs its token use to LLM_LOG (if set) so cost can be totalled."""
import json
import os
import time
import urllib.request

MODEL = "gemini-2.5-flash-lite"
URL = "https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent"
PRICE_IN, PRICE_OUT = 0.10 / 1e6, 0.40 / 1e6  # USD per token (list price)


def call(prompt: str, system: str = "", max_tokens: int = 8192, json_out: bool = True, temperature: float = 0.0) -> str:
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": max_tokens, "temperature": temperature,
                                 "thinkingConfig": {"thinkingBudget": 0}}}
    if json_out:
        body["generationConfig"]["responseMimeType"] = "application/json"
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    req = urllib.request.Request(URL.format(m=MODEL), data=json.dumps(body).encode(),
                                 headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"], "content-type": "application/json"})
    err = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=180) as f:
                d = json.loads(f.read())
            u = d.get("usageMetadata", {})
            if os.environ.get("LLM_LOG"):
                with open(os.environ["LLM_LOG"], "a") as lf:
                    lf.write(json.dumps({"in": u.get("promptTokenCount", 0), "out": u.get("candidatesTokenCount", 0)}) + "\n")
            return "".join(p.get("text", "") for p in d["candidates"][0]["content"]["parts"])
        except Exception as e:  # noqa: BLE001  (rate limit / transient)
            err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"LLM call failed: {err!r}")


def cost(log_path: str) -> float:
    t_in = t_out = 0
    for l in open(log_path):
        j = json.loads(l)
        t_in, t_out = t_in + j["in"], t_out + j["out"]
    return t_in * PRICE_IN + t_out * PRICE_OUT

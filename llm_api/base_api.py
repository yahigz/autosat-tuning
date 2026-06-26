import os
import json
import time

import openai
from prompting import build_structured_schema, build_tool_schema, parse_structured_response

_JSON_OBJECT_FORMAT = {"type": "json_object"}


def _parse_structured_response(content: str):
    try:
        data = json.loads(content)
        return str(data.get("code", "")).strip(), str(data.get("reason", "")).strip()
    except Exception as exc:
        print(f"[StructuredOutput] JSON parse error: {exc}. Raw:\n{content[:300]}", flush=True)
        return "", ""


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseCallAPI:
    def __init__(self, api_base, api_key, model_name):
        self.api_base   = api_base
        self.api_key    = api_key
        self.model_name = model_name
        self._cum_prompt_tokens     = 0
        self._cum_completion_tokens = 0
        self._cum_total_tokens      = 0
        self._cum_calls             = 0
        self._retry_sleep_seconds   = max(1, int(os.getenv("AUTOSAT_API_RETRY_SECONDS", "10")))
        self._max_retries           = int(os.getenv("AUTOSAT_API_MAX_RETRIES", "0"))
        self._structured_output     = (
            os.getenv("AUTOSAT_STRUCTURED_OUTPUT", "1").strip()
            not in ("0", "false", "False", "no")
        )
        self._config_payload = {}

    def _make_client(self) -> openai.OpenAI:
        kwargs: dict = {"api_key": self.api_key or "sk-placeholder"}
        if self.api_base:
            kwargs["base_url"] = self.api_base
        return openai.OpenAI(**kwargs)

    def _log_token_usage(self, usage):
        if usage is None:
            return
        # usage may be an object or a dict
        if isinstance(usage, dict):
            p = usage.get("prompt_tokens", 0)
            c = usage.get("completion_tokens", 0)
            t = usage.get("total_tokens", 0)
        else:
            p = getattr(usage, "prompt_tokens", 0) or 0
            c = getattr(usage, "completion_tokens", 0) or 0
            t = getattr(usage, "total_tokens", 0) or 0
        try:
            p, c, t = int(p), int(c), int(t)
        except Exception:
            return
        self._cum_calls             += 1
        self._cum_prompt_tokens     += p
        self._cum_completion_tokens += c
        self._cum_total_tokens      += t
        print(
            f"[TokenUsage] model={self.model_name} "
            f"call_prompt={p} call_completion={c} call_total={t} "
            f"cum_calls={self._cum_calls} cum_prompt={self._cum_prompt_tokens} "
            f"cum_completion={self._cum_completion_tokens} cum_total={self._cum_total_tokens}",
            flush=True,
        )

    def load_prompt(self, file_dir):
        with open(file_dir, "r", encoding="utf-8") as f:
            return f.read()

    def _call_with_retries(self, request_fn, description="API call"):
        attempt = 0
        while True:
            try:
                return request_fn()
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                attempt += 1
                if self._max_retries > 0 and attempt > self._max_retries:
                    print(f"[{description}] failed after {attempt-1} retries: {exc}", flush=True)
                    raise
                print(
                    f"[{description}] error on attempt {attempt}: {exc}. "
                    f"Retrying in {self._retry_sleep_seconds} seconds...",
                    flush=True,
                )
                time.sleep(self._retry_sleep_seconds)

    def _build_structured_system_prompt(self):
        return (
            "You are an expert C++ SAT solver engineer. "
            "You MUST respond with a valid JSON object matching the required schema. "
            "The 'code' field must contain only the C++ function body (no markdown fences). "
            "The 'title' field must be a concise change name. "
            "The 'reason' field is a brief motivation."
        )

    def call_api(self, prompt_file, temperature=1.0):
        raise NotImplementedError

    def call_api_structured(self, prompt_file: str, temperature: float = 1.0, task_name: str = ""):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# GPT / OpenAI-compatible (openai >= 1.0)
# ---------------------------------------------------------------------------

class GPTCallAPI(BaseCallAPI):
    def call_api(self, prompt_file, temperature=1.0):
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt = f.read()
        if self._structured_output:
            return self._call_structured_raw(prompt, temperature)
        client = self._make_client()
        response = self._call_with_retries(
            lambda: client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a chatbot"},
                    {"role": "user",   "content": prompt},
                ],
                temperature=temperature,
            ),
            description="GPTCallAPI.call_api",
        )
        self._log_token_usage(response.usage)
        return response.choices[0].message.content

    def _call_structured_raw(self, prompt: str, temperature: float, task_name: str = "") -> str:
        client     = self._make_client()
        system_msg = self._build_structured_system_prompt()
        schema     = build_structured_schema(self._config_payload, task_name=task_name)

        def _with_schema():
            return client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user",   "content": prompt},
                ],
                temperature=temperature,
                response_format=schema,
            )

        def _json_object():
            return client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user",   "content": prompt},
                ],
                temperature=temperature,
                response_format=_JSON_OBJECT_FORMAT,
            )

        for attempt_fn, label in [(_with_schema, "json_schema"), (_json_object, "json_object")]:
            try:
                resp = self._call_with_retries(attempt_fn, description=f"GPTCallAPI.structured[{label}]")
                self._log_token_usage(resp.usage)
                content = resp.choices[0].message.content
                code, _ = _parse_structured_response(content)
                if code:
                    print(f"[StructuredOutput] Success via {label}", flush=True)
                    return content
                print(f"[StructuredOutput] Empty code via {label}, trying next.", flush=True)
            except Exception as exc:
                print(f"[StructuredOutput] {label} failed: {exc}. Trying next.", flush=True)

        # plain text fallback
        print("[StructuredOutput] Falling back to plain text.", flush=True)
        resp = self._call_with_retries(
            lambda: client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a chatbot"},
                    {"role": "user",   "content": prompt},
                ],
                temperature=temperature,
            ),
            description="GPTCallAPI.call_api[fallback]",
        )
        self._log_token_usage(resp.usage)
        return resp.choices[0].message.content

    def call_api_structured(self, prompt_file: str, temperature: float = 1.0, task_name: str = ""):
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt = f.read()
        raw = self._call_structured_raw(prompt, temperature, task_name=task_name)
        return parse_structured_response(raw)

    def call_structured_all(self, prompt: str, temperature: float, schema) -> str:
        """Multi-task structured call using OpenAI json_schema."""
        client = self._make_client()
        try:
            resp = self._call_with_retries(
                lambda: client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self._build_structured_system_prompt()},
                        {"role": "user",   "content": prompt},
                    ],
                    temperature=temperature,
                    response_format=schema,
                ),
                description="GPTCallAPI.structured_all",
            )
            self._log_token_usage(resp.usage)
            return resp.choices[0].message.content
        except Exception as exc:
            print(f"[all-tasks][openai] structured failed: {exc}, falling back to plain text", flush=True)
            resp = self._call_with_retries(
                lambda: client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a chatbot"},
                        {"role": "user",   "content": prompt},
                    ],
                    temperature=temperature,
                ),
                description="GPTCallAPI.structured_all[fallback]",
            )
            self._log_token_usage(resp.usage)
            return resp.choices[0].message.content


# ---------------------------------------------------------------------------
# Local / vLLM (openai-compatible, openai >= 1.0)
# ---------------------------------------------------------------------------

class LocalCallAPI(BaseCallAPI):
    def call_api(self, prompt_file, temperature=1.0):
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt = f.read()
        client = self._make_client()
        response = self._call_with_retries(
            lambda: client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a chatbot"},
                    {"role": "user",   "content": prompt},
                ],
                stop=["<|im_end|>"],
            ),
            description="LocalCallAPI.call_api",
        )
        self._log_token_usage(response.usage)
        return response.choices[0].message.content

    def call_api_structured(self, prompt_file: str, temperature: float = 1.0, task_name: str = ""):
        content = self.call_api(prompt_file, temperature)
        return parse_structured_response(content)


# ---------------------------------------------------------------------------
# Eliza API  (internal Yandex gateway — native Anthropic Messages protocol)
# ---------------------------------------------------------------------------

class ElizaCallAPI(BaseCallAPI):
    """
    Client for the Eliza LLM gateway (raw HTTP, no anthropic SDK).

    Set in .env:
        AUTOSAT_API_TYPE=eliza
        AUTOSAT_API_KEY=<soy_oauth_token>
        AUTOSAT_LLM_MODEL=claude-sonnet-4-6
    """

    _MESSAGES_URL  = "https://api.eliza.yandex.net/anthropic/v1/messages"
    _DEFAULT_MAX_TOKENS = 4096

    def __init__(self, api_base: str, api_key: str, model_name: str):
        super().__init__(api_base, api_key, model_name)
        self._soy_token = api_key.strip()
        if not self._soy_token:
            raise ValueError("ElizaCallAPI requires a SOY OAuth token (AUTOSAT_API_KEY).")
        try:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        except Exception:
            pass

    def _make_headers(self) -> dict:
        return {
            "authorization": f"OAuth {self._soy_token} ",
            "content-type":  "application/json",
        }

    def _post(self, payload: dict) -> dict:
        import requests as _requests
        resp = _requests.post(
            self._MESSAGES_URL,
            json=payload,
            headers=self._make_headers(),
            verify=False,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        if "response" in data and isinstance(data["response"], dict):
            return data["response"]
        return data

    def _extract_text(self, data: dict) -> str:
        return "\n".join(
            b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"
        )

    def _log_usage_from_response(self, data: dict):
        u = data.get("usage", {})
        self._log_token_usage({
            "prompt_tokens":     u.get("input_tokens",  0),
            "completion_tokens": u.get("output_tokens", 0),
            "total_tokens":      u.get("input_tokens",  0) + u.get("output_tokens", 0),
        })

    def _call_plain(self, system_msg: str, user_msg: str, temperature: float) -> str:
        payload = {
            "model":      self.model_name,
            "max_tokens": self._DEFAULT_MAX_TOKENS,
            "system":     system_msg,
            "messages":   [{"role": "user", "content": user_msg}],
            "temperature": temperature,
        }
        data = self._call_with_retries(lambda: self._post(payload), description="ElizaCallAPI[plain]")
        self._log_usage_from_response(data)
        return self._extract_text(data)

    def _call_structured_raw(self, prompt: str, temperature: float, task_name: str = "") -> str:
        tool_schema = build_tool_schema(self._config_payload, task_name=task_name)
        payload = {
            "model":       self.model_name,
            "max_tokens":  self._DEFAULT_MAX_TOKENS,
            "system":      self._build_structured_system_prompt(),
            "messages":    [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "tools":       [tool_schema],
            "tool_choice": {"type": "any"},
        }
        try:
            data = self._call_with_retries(lambda: self._post(payload), description="ElizaCallAPI[structured]")
            self._log_usage_from_response(data)
            for block in data.get("content", []):
                if block.get("type") == "tool_use":
                    return json.dumps(block.get("input", {}))
            print("[StructuredOutput][Eliza] No tool_use block, falling back.", flush=True)
        except Exception as exc:
            print(f"[StructuredOutput][Eliza] failed: {exc}. Falling back.", flush=True)
        return self._call_plain("You are a chatbot", prompt, temperature)

    def call_api(self, prompt_file: str, temperature: float = 1.0) -> str:
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt = f.read()
        if self._structured_output:
            return self._call_structured_raw(prompt, temperature)
        return self._call_plain("You are a chatbot", prompt, temperature)

    def call_api_structured(self, prompt_file: str, temperature: float = 1.0, task_name: str = ""):
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt = f.read()
        raw = self._call_structured_raw(prompt, temperature, task_name=task_name)
        return parse_structured_response(raw)

    def call_structured_all(self, prompt: str, temperature: float, tool_schema: dict) -> str:
        """Multi-task structured call for all-mode using Eliza tool_use."""
        payload = {
            "model":       self.model_name,
            "max_tokens":  self._DEFAULT_MAX_TOKENS,
            "system":      self._build_structured_system_prompt(),
            "messages":    [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "tools":       [tool_schema],
            "tool_choice": {"type": "any"},
        }
        try:
            data = self._call_with_retries(lambda: self._post(payload), description="ElizaCallAPI[all]")
            self._log_usage_from_response(data)
            for block in data.get("content", []):
                if block.get("type") == "tool_use":
                    return json.dumps(block.get("input", {}))
            print("[all-tasks][Eliza] No tool_use block, falling back.", flush=True)
        except Exception as exc:
            print(f"[all-tasks][Eliza] failed: {exc}. Falling back.", flush=True)
        return self._call_plain("You are a chatbot", prompt, temperature)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_llm_api(args):
    model_name = str(getattr(args, "llm_model", "") or "").strip()
    api_base   = str(getattr(args, "api_base",  "") or "").strip()
    api_key    = str(getattr(args, "api_key",   "") or "").strip()
    api_type   = str(os.getenv("AUTOSAT_API_TYPE", "") or "").strip().lower()

    if api_type == "eliza":
        resolved = model_name.removeprefix("eliza/") if model_name.startswith("eliza/") else model_name
        if not resolved:
            raise ValueError("Eliza API requires a model name (AUTOSAT_LLM_MODEL or llm_model in config).")
        if not api_key:
            raise ValueError("Eliza API requires a SOY OAuth token (AUTOSAT_API_KEY).")
        print(f"[LLM] ElizaCallAPI model={resolved}", flush=True)
        api = ElizaCallAPI(api_base="", api_key=api_key, model_name=resolved)
        api._config_payload = dict(getattr(args, "_loaded_config_payload", {}) or {})
        return api

    # Resolve api_base from env if needed
    if not api_base:
        api_base = (os.getenv("AUTOSAT_API_BASE") or os.getenv("DEEPINFRA_API_BASE") or "").strip()

    if api_base or model_name.startswith("gpt-"):
        print(f"[LLM] GPTCallAPI model={model_name} base={api_base or '(openai)'}", flush=True)
        api = GPTCallAPI(api_base=api_base, api_key=api_key, model_name=model_name)
        api._config_payload = dict(getattr(args, "_loaded_config_payload", {}) or {})
        return api

    _LOCAL = {
        "Qwen":     ("http://172.26.1.16:31251/v1", "modelscope/qwen/Qwen-72B-Chat"),
        "llama":    ("http://172.26.1.16:31251/v1", "modelscope/modelscope/Llama-2-70b-chat-ms"),
        "deepseek": ("http://172.26.1.16:31251/v1", "modelscope/deepseek-ai/deepseek-coder-33b-instruct"),
    }
    if model_name in _LOCAL:
        base, name = _LOCAL[model_name]
        print(f"[LLM] LocalCallAPI model={name}", flush=True)
        api = LocalCallAPI(api_base=base, api_key="sk-", model_name=name)
        api._config_payload = dict(getattr(args, "_loaded_config_payload", {}) or {})
        return api

    raise NotImplementedError(
        f"Cannot resolve LLM backend for model={model_name!r}. "
        "Set AUTOSAT_API_TYPE=eliza, or set AUTOSAT_API_BASE+AUTOSAT_API_KEY for OpenAI-compatible."
    )

"""Budgeted Anthropic-compatible QGenie adapter for the matched study.

Complete returns text like the existing ModelClient. Requests and responses are
cached without credentials; cache hits never count as additional model samples.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from src.constants import MODEL_DEFAULT


class StudyStopped(RuntimeError):
    """A credential, transport, response or budget problem stopped inference."""


def credential(key_file: Path | None) -> str:
    key = os.environ.get('QGENIE_API_KEY') or os.environ.get('ANTHROPIC_API_KEY')
    if key:
        return key.strip()
    if key_file is None or not key_file.is_file():
        raise StudyStopped('QGenie key is not configured.')
    if stat.S_IMODE(key_file.stat().st_mode) & 0o077:
        raise StudyStopped('The key file must have permissions 600.')
    key = key_file.read_text().strip()
    if not key:
        raise StudyStopped('The key file is empty.')
    return key


def http_transport(url: str, headers: dict, body: dict, timeout: float) -> dict:
    request = Request(url, data=json.dumps(body).encode(), headers=headers, method='POST')
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except HTTPError as exc:
        # Do not persist provider bodies, headers, credential values or URLs in errors.
        raise StudyStopped(f'Provider returned HTTP {exc.code}.') from None
    except (URLError, TimeoutError, OSError, ValueError):
        raise StudyStopped('Provider transport failed or returned invalid JSON.') from None


class StudyClient:
    def __init__(self, output: Path, key: str, *, model: str = MODEL_DEFAULT,
                 base_url: str = 'https://qgenie-api.qualcomm.com', max_calls: int = 110,
                 max_minutes: float = 45, transport=http_transport, now=time.monotonic):
        parsed = urlparse(base_url)
        if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.query or parsed.fragment:
            raise ValueError('Use an HTTPS provider base URL without credentials or query parameters.')
        if max_calls < 1 or max_minutes <= 0:
            raise ValueError('Call and time limits must be positive.')
        self.output, self.key, self.model = output, key, model
        self.base_url = base_url.rstrip('/')
        self.max_calls, self.transport, self.now = max_calls, transport, now
        self.deadline = now() + max_minutes * 60
        self.context = {'purpose': 'task', 'max_tokens': 512}
        self.log = output / 'hosted/api_calls.jsonl'
        self.cache_dir = output / 'hosted/cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.records = [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []
        self.cache_events = []
        self.last = {}

    def set_context(self, **values) -> None:
        self.context = {'purpose': 'task', 'max_tokens': 512, **values}

    def _record(self, row: dict) -> None:
        self.records.append(row)
        self.log.parent.mkdir(parents=True, exist_ok=True)
        with self.log.open('a') as stream:
            stream.write(json.dumps(row, sort_keys=True) + '\n')

    def complete(self, prompt: str, model: str | None = None) -> str:
        body = {'model': model or self.model, 'temperature': 0.0,
                'max_tokens': self.context['max_tokens'],
                'messages': [{'role': 'user', 'content': prompt}]}
        fingerprint = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        cached = self.cache_dir / f'{fingerprint}.json'
        if cached.exists():
            row = json.loads(cached.read_text())
            self.last = {**row, **self.context, 'cache_hit': True}
            self.cache_events.append({k: v for k, v in self.last.items() if k != 'text'})
            return row['text']
        if len(self.records) >= self.max_calls:
            raise StudyStopped('The API request limit was reached.')
        remaining = self.deadline - self.now()
        if remaining <= 0:
            raise StudyStopped('The inference time limit was reached.')
        started = self.now()
        common = {**self.context, 'prompt_sha256': fingerprint, 'model': body['model'],
                  'temperature': 0.0, 'attempt': len(self.records) + 1}
        try:
            response = self.transport(self.base_url + '/v1/messages',
                                      {'Content-Type': 'application/json', 'x-api-key': self.key,
                                       'anthropic-version': '2023-06-01'}, body, min(30, remaining))
            content = response.get('content', [])
            text = '\n'.join(block['text'] for block in content if block.get('type') == 'text')
            if not text or response.get('stop_reason') == 'max_tokens':
                raise StudyStopped('Provider returned empty or truncated content.')
        except StudyStopped as exc:
            self._record({**common, 'status': 'failed', 'elapsed_seconds': self.now() - started,
                          'error': str(exc)})
            raise
        usage = response.get('usage', {})
        row = {**common, 'status': 'ok', 'elapsed_seconds': self.now() - started,
               'response_id': response.get('id'), 'response_model': response.get('model'),
               'input_tokens': usage.get('input_tokens'), 'output_tokens': usage.get('output_tokens'),
               'cache_read_input_tokens': usage.get('cache_read_input_tokens'),
               'cache_creation_input_tokens': usage.get('cache_creation_input_tokens'),
               'text': text}
        cached.write_text(json.dumps(row, indent=2) + '\n')
        self._record({k: v for k, v in row.items() if k != 'text'})
        self.last = {**row, 'cache_hit': False}
        return text

    def preflight(self) -> None:
        self.set_context(purpose='preflight', max_tokens=5)
        self.complete('Reply with OK.')

    def save_cache_events(self) -> None:
        path = self.output / 'hosted/cache_hits.json'
        previous = json.loads(path.read_text()) if path.exists() else []
        path.write_text(json.dumps(previous + self.cache_events, indent=2) + '\n')
        self.cache_events.clear()

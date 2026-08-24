from __future__ import annotations

import base64
import hashlib
import hmac
import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

from facehide.config import MessageChannel
from facehide.i18n import t

JsonFn = Callable[..., tuple[int, str]]

_token_lock = threading.Lock()
_token_cache: dict[str, tuple[str, float]] = {}


@dataclass
class NotifyEvent:
    person: str
    score: float
    when: datetime
    test: bool = False


def format_when(when: datetime) -> str:
    return when.strftime("%Y-%m-%d %H:%M:%S")


def render_text(event: NotifyEvent, keyword: str = "") -> str:
    title = t("notify.test_title") if event.test else t("notify.alert_title", name=event.person)
    body = t("notify.body", time=format_when(event.when), score=event.score)
    parts = [keyword.strip(), title, body]
    return "\n".join(part for part in parts if part)


def dingtalk_sign(secret: str, timestamp: str) -> str:
    raw = f"{timestamp}\n{secret}"
    digest = hmac.new(secret.encode("utf-8"), raw.encode("utf-8"), hashlib.sha256).digest()
    return urllib.parse.quote_plus(base64.b64encode(digest))


def feishu_sign(secret: str, timestamp: str) -> str:
    raw = f"{timestamp}\n{secret}"
    digest = hmac.new(raw.encode("utf-8"), digestmod=hashlib.sha256).digest()
    return base64.b64encode(digest).decode("utf-8")


def apply_vars(text: str, event: NotifyEvent, message: str) -> str:
    return (
        text.replace("{person}", event.person)
        .replace("{score}", f"{event.score:.2f}")
        .replace("{time}", format_when(event.when))
        .replace("{message}", message)
        .replace("{event}", "blacklist_test" if event.test else "blacklist")
    )


def request_json(
    method: str,
    url: str,
    body: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 10.0,
) -> tuple[int, str]:
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method.upper())
    req.add_header("Content-Type", "application/json; charset=utf-8")
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = resp.read().decode("utf-8", errors="replace")
            return int(getattr(resp, "status", 200) or 200), payload
    except urllib.error.HTTPError as exc:
        payload = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
        return int(exc.code), payload


def _ok_payload(status: int, payload: str) -> bool:
    if status >= 400:
        return False
    text = payload.strip()
    if not text:
        return True
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return True
    if not isinstance(data, dict):
        return True
    if "errcode" in data:
        return int(data.get("errcode") or 0) == 0
    if "StatusCode" in data:
        return int(data.get("StatusCode") or 0) == 0
    if "code" in data:
        return str(data.get("code")) in {"0", "ok", "success"}
    return True


def _dingtalk_token(channel: MessageChannel, http: JsonFn) -> str:
    key = f"{channel.app_key}:{channel.app_secret}"
    now = time.time()
    with _token_lock:
        cached = _token_cache.get(key)
        if cached and cached[1] > now + 60:
            return cached[0]
    query = urllib.parse.urlencode({"appkey": channel.app_key, "appsecret": channel.app_secret})
    status, payload = http("GET", f"https://oapi.dingtalk.com/gettoken?{query}", None, None)
    data = json.loads(payload or "{}")
    token = str(data.get("access_token") or "")
    if status >= 400 or int(data.get("errcode") or 0) != 0 or not token:
        raise RuntimeError(payload or t("notify.http_fail", status=status))
    expires = now + max(60.0, float(data.get("expires_in") or 7200) - 120)
    with _token_lock:
        _token_cache[key] = (token, expires)
    return token


def send_dingtalk_group(channel: MessageChannel, event: NotifyEvent, http: JsonFn) -> str:
    keyword = channel.keyword if channel.auth_mode == "keyword" else ""
    text = render_text(event, keyword)
    url = channel.webhook.strip()
    if not url:
        raise RuntimeError(t("notify.need_webhook"))
    if channel.auth_mode == "sign":
        if not channel.secret.strip():
            raise RuntimeError(t("notify.need_secret"))
        timestamp = str(round(time.time() * 1000))
        sign = dingtalk_sign(channel.secret.strip(), timestamp)
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}timestamp={timestamp}&sign={sign}"
    status, payload = http("POST", url, {"msgtype": "text", "text": {"content": text}}, None)
    if not _ok_payload(status, payload):
        raise RuntimeError(payload or t("notify.http_fail", status=status))
    return t("notify.sent", name=channel.name)


def send_dingtalk_app(channel: MessageChannel, event: NotifyEvent, http: JsonFn) -> str:
    if not channel.app_key.strip() or not channel.app_secret.strip():
        raise RuntimeError(t("notify.need_app"))
    if not channel.conversation_id.strip():
        raise RuntimeError(t("notify.need_conversation"))
    token = _dingtalk_token(channel, http)
    robot = channel.robot_code.strip() or channel.app_key.strip()
    text = render_text(event, channel.keyword)
    body = {
        "robotCode": robot,
        "openConversationId": channel.conversation_id.strip(),
        "msgKey": "sampleText",
        "msgParam": json.dumps({"content": text}, ensure_ascii=False),
    }
    status, payload = http(
        "POST",
        "https://api.dingtalk.com/v1.0/robot/groupMessages/send",
        body,
        {"x-acs-dingtalk-access-token": token},
    )
    if not _ok_payload(status, payload):
        raise RuntimeError(payload or t("notify.http_fail", status=status))
    return t("notify.sent", name=channel.name)


def send_feishu(channel: MessageChannel, event: NotifyEvent, http: JsonFn) -> str:
    url = channel.webhook.strip()
    if not url:
        raise RuntimeError(t("notify.need_webhook"))
    text = render_text(event, channel.keyword)
    body: dict[str, Any] = {"msg_type": "text", "content": {"text": text}}
    if channel.secret.strip():
        timestamp = str(int(time.time()))
        body["timestamp"] = timestamp
        body["sign"] = feishu_sign(channel.secret.strip(), timestamp)
    status, payload = http("POST", url, body, None)
    if not _ok_payload(status, payload):
        raise RuntimeError(payload or t("notify.http_fail", status=status))
    return t("notify.sent", name=channel.name)


def send_webhook(channel: MessageChannel, event: NotifyEvent, http: JsonFn) -> str:
    url = channel.url.strip() or channel.webhook.strip()
    if not url:
        raise RuntimeError(t("notify.need_url"))
    message = render_text(event)
    payload: dict[str, Any] = {
        "event": "blacklist_test" if event.test else "blacklist",
        "person": event.person,
        "score": round(event.score, 4),
        "time": format_when(event.when),
        "message": message,
    }
    for item in channel.params:
        payload[apply_vars(item.key, event, message)] = apply_vars(item.value, event, message)
    headers = {apply_vars(item.key, event, message): apply_vars(item.value, event, message) for item in channel.headers}
    method = (channel.method or "POST").upper()
    if method == "GET":
        query = urllib.parse.urlencode({str(key): str(value) for key, value in payload.items()})
        sep = "&" if "?" in url else "?"
        status, body = http("GET", f"{url}{sep}{query}", None, headers)
    else:
        status, body = http("POST", url, payload, headers)
    if status >= 400:
        raise RuntimeError(body or t("notify.http_fail", status=status))
    return t("notify.sent", name=channel.name)


_SENDERS = {
    "dingtalk_group": send_dingtalk_group,
    "dingtalk_app": send_dingtalk_app,
    "feishu": send_feishu,
    "webhook": send_webhook,
}


def send_channel(channel: MessageChannel, event: NotifyEvent, http: JsonFn | None = None) -> str:
    sender = _SENDERS.get(channel.kind)
    if sender is None:
        raise RuntimeError(t("notify.unknown_kind", kind=channel.kind))
    return sender(channel, event, http or request_json)


def dispatch(channels: list[MessageChannel], event: NotifyEvent, http: JsonFn | None = None) -> list[str]:
    lines: list[str] = []
    for channel in channels:
        if not channel.enabled:
            continue
        try:
            lines.append(send_channel(channel, event, http))
        except Exception as exc:  # noqa: BLE001
            lines.append(t("notify.fail", name=channel.name, error=exc))
    if not lines:
        lines.append(t("notify.no_channel"))
    return lines

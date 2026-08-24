import unittest
from datetime import datetime
from urllib.parse import parse_qs, urlparse

from facehide.config import KvPair, MessageChannel
from facehide.notify import (
    NotifyEvent,
    apply_vars,
    dingtalk_sign,
    dispatch,
    feishu_sign,
    render_text,
    send_channel,
)


class NotifyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.event = NotifyEvent(person="Ada", score=0.86, when=datetime(2026, 8, 24, 15, 30, 12))

    def test_dingtalk_sign_is_stable(self) -> None:
        sign = dingtalk_sign("SECabc", "1710000000000")
        self.assertTrue(sign)
        self.assertNotIn("+", sign)
        self.assertEqual(sign, dingtalk_sign("SECabc", "1710000000000"))

    def test_feishu_sign_is_stable(self) -> None:
        sign = feishu_sign("secret", "1710000000")
        self.assertEqual(sign, feishu_sign("secret", "1710000000"))

    def test_keyword_is_prepended(self) -> None:
        text = render_text(self.event, "告警")
        self.assertTrue(text.startswith("告警"))
        self.assertIn("Ada", text)

    def test_dingtalk_group_sign_appends_query(self) -> None:
        calls: list[tuple] = []

        def http(method, url, body, headers):
            calls.append((method, url, body, headers))
            return 200, '{"errcode":0}'

        channel = MessageChannel(
            id="1",
            kind="dingtalk_group",
            name="群",
            webhook="https://oapi.dingtalk.com/robot/send?access_token=x",
            auth_mode="sign",
            secret="SECabc",
        )
        send_channel(channel, self.event, http)
        self.assertEqual(calls[0][0], "POST")
        parsed = urlparse(calls[0][1])
        query = parse_qs(parsed.query)
        self.assertIn("timestamp", query)
        self.assertIn("sign", query)
        self.assertEqual(calls[0][2]["msgtype"], "text")

    def test_dingtalk_group_keyword_and_ip(self) -> None:
        seen: list[str] = []

        def http(method, url, body, headers):
            seen.append(body["text"]["content"])
            return 200, '{"errcode":0}'

        keyword = MessageChannel(
            id="k",
            kind="dingtalk_group",
            name="k",
            webhook="https://example/hook",
            auth_mode="keyword",
            keyword="告警",
        )
        ip = MessageChannel(
            id="ip",
            kind="dingtalk_group",
            name="ip",
            webhook="https://example/hook",
            auth_mode="ip",
        )
        send_channel(keyword, self.event, http)
        send_channel(ip, self.event, http)
        self.assertTrue(seen[0].startswith("告警"))
        self.assertFalse(seen[1].startswith("告警"))

    def test_webhook_merges_params_and_headers(self) -> None:
        calls: list[tuple] = []

        def http(method, url, body, headers):
            calls.append((method, url, body, headers))
            return 200, ""

        channel = MessageChannel(
            id="w",
            kind="webhook",
            name="cb",
            url="https://example.com/hook",
            headers=[KvPair("X-Token", "abc")],
            params=[KvPair("source", "facehide"), KvPair("who", "{person}")],
        )
        send_channel(channel, self.event, http)
        method, url, body, headers = calls[0]
        self.assertEqual(method, "POST")
        self.assertEqual(body["person"], "Ada")
        self.assertEqual(body["source"], "facehide")
        self.assertEqual(body["who"], "Ada")
        self.assertEqual(headers["X-Token"], "abc")
        self.assertEqual(apply_vars("{event}", self.event, "m"), "blacklist")

    def test_dispatch_skips_disabled_and_collects_errors(self) -> None:
        def http(method, url, body, headers):
            raise RuntimeError("boom")

        enabled = MessageChannel(id="a", kind="feishu", name="飞书", webhook="https://example/f")
        disabled = MessageChannel(id="b", kind="feishu", name="关", enabled=False, webhook="https://example/f")
        lines = dispatch([disabled, enabled], self.event, http)
        self.assertEqual(len(lines), 1)
        self.assertIn("boom", lines[0])


if __name__ == "__main__":
    unittest.main()

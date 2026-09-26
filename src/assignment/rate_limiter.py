"""
Assignment 11 — Rate Limiter.

Sliding-window, per-user rate limiting. Blocks abuse that other
guardrail layers do not address (flooding / cost attacks).
"""
from __future__ import annotations

from collections import defaultdict, deque
import time

from google.adk.plugins import base_plugin
from google.genai import types


class RateLimitPlugin(base_plugin.BasePlugin):
    """Block users who exceed max_requests within window_seconds."""

    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        super().__init__(name="rate_limiter")
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.user_windows: dict[str, deque] = defaultdict(deque)
        self.blocked_count = 0
        self.total_count = 0
        self.last_block_reason: str | None = None

    def _block_response(self, message: str) -> types.Content:
        return types.Content(
            role="model",
            parts=[types.Part.from_text(text=message)],
        )

    async def on_user_message_callback(self, *, invocation_context, user_message):
        """Return Content to block, or None to allow."""
        self.total_count += 1
        self.last_block_reason = None
        user_id = getattr(invocation_context, "user_id", None) or "anonymous"
        # monotonic: không bị ảnh hưởng khi đồng hồ hệ thống bị chỉnh
        now = time.monotonic()
        window = self.user_windows[user_id]

        # 1. Bỏ các timestamp đã trượt ra khỏi cửa sổ [now - window_seconds, now]
        while window and window[0] <= now - self.window_seconds:
            window.popleft()

        # 2. Hết quota → chặn. Request bị chặn KHÔNG được ghi vào window,
        #    nên spam liên tục không làm thời gian chờ kéo dài vô hạn.
        if len(window) >= self.max_requests:
            wait = self.window_seconds - (now - window[0])
            self.blocked_count += 1
            self.last_block_reason = "rate_limited"
            return self._block_response(
                f"Rate limit exceeded. Try again in {wait:.0f}s. "
                f"(Tối đa {self.max_requests} yêu cầu / {self.window_seconds}s.)"
            )

        # 3. Còn quota → ghi nhận và cho qua
        window.append(now)
        return None

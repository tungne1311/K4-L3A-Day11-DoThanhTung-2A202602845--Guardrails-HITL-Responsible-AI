"""
Assignment 11 — Monitoring & Alerts.

Tracks block rate, rate-limit hits, judge fail rate.
Fires alerts when thresholds are exceeded.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


def default_metrics_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "metrics.json")


@dataclass
class Alert:
    metric: str
    value: float
    threshold: float
    message: str


@dataclass
class MonitoringAlert:
    """Aggregate counters from pipeline plugins and emit alerts."""

    block_rate_threshold: float = 0.5
    rate_limit_hit_threshold: int = 5
    judge_fail_rate_threshold: float = 0.3
    # ≥1 lần LLM sinh ra secret (bị output guardrail che) = lớp input đã bị vượt qua
    secret_redaction_threshold: int = 1
    alerts: list[Alert] = field(default_factory=list)

    # Counters — update these from your pipeline after each request
    total_requests: int = 0
    blocked_requests: int = 0
    rate_limit_hits: int = 0
    judge_checks: int = 0
    judge_fails: int = 0
    input_blocks: int = 0
    output_redactions: int = 0
    secret_redactions: int = 0
    llm_errors: int = 0

    def record(
        self,
        *,
        blocked: bool,
        layer: str | None,
        redacted: bool = False,
        secret_redacted: bool = False,
        judge_checked: bool = False,
        judge_failed: bool = False,
        llm_error: bool = False,
    ) -> None:
        """Cập nhật bộ đếm sau mỗi request (pipeline gọi hàm này)."""
        self.total_requests += 1
        self.blocked_requests += int(blocked)
        self.rate_limit_hits += int(layer == "rate_limiter")
        self.input_blocks += int(blocked and layer == "input_guardrail")
        self.output_redactions += int(redacted)
        self.secret_redactions += int(secret_redacted)
        self.judge_checks += int(judge_checked)
        self.judge_fails += int(judge_failed)
        self.llm_errors += int(llm_error)

    def check_metrics(self) -> list[Alert]:
        """Compute rates, rebuild ``self.alerts`` for every threshold reached."""
        snap = self.snapshot()
        alerts: list[Alert] = []

        if self.total_requests and snap["block_rate"] >= self.block_rate_threshold:
            alerts.append(Alert(
                "block_rate", round(snap["block_rate"], 3), self.block_rate_threshold,
                "Tỉ lệ chặn cao — có thể đang có đợt tấn công (hoặc filter chặn nhầm).",
            ))
        if self.rate_limit_hits >= self.rate_limit_hit_threshold:
            alerts.append(Alert(
                "rate_limit_hits", self.rate_limit_hits, self.rate_limit_hit_threshold,
                "Nhiều request bị rate limit — có thể đang bị spam / flooding.",
            ))
        if self.judge_checks and snap["judge_fail_rate"] >= self.judge_fail_rate_threshold:
            alerts.append(Alert(
                "judge_fail_rate", round(snap["judge_fail_rate"], 3),
                self.judge_fail_rate_threshold,
                "LLM-as-Judge đánh UNSAFE nhiều — kiểm tra lại model / prompt.",
            ))
        if self.secret_redactions >= self.secret_redaction_threshold:
            alerts.append(Alert(
                "secret_redactions", self.secret_redactions, self.secret_redaction_threshold,
                "LLM đã sinh ra dữ liệu mật (đã bị che ở output) — input guardrail bị vượt qua.",
            ))

        # Tính lại từ đầu mỗi lần gọi → không bị nhân đôi alert
        self.alerts = alerts
        return alerts

    def export_json(self, filepath: str | None = None):
        """Write metrics + alerts to JSON under repo-root ``outputs/`` by default."""
        self.check_metrics()
        path = Path(filepath or default_metrics_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "thresholds": {
                "block_rate": self.block_rate_threshold,
                "rate_limit_hits": self.rate_limit_hit_threshold,
                "judge_fail_rate": self.judge_fail_rate_threshold,
                "secret_redactions": self.secret_redaction_threshold,
            },
            **self.snapshot(),
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return str(path)

    def snapshot(self) -> dict:
        block_rate = (
            self.blocked_requests / self.total_requests
            if self.total_requests
            else 0.0
        )
        judge_fail_rate = (
            self.judge_fails / self.judge_checks if self.judge_checks else 0.0
        )
        return {
            "total_requests": self.total_requests,
            "blocked_requests": self.blocked_requests,
            "block_rate": block_rate,
            "rate_limit_hits": self.rate_limit_hits,
            "input_blocks": self.input_blocks,
            "output_redactions": self.output_redactions,
            "secret_redactions": self.secret_redactions,
            "llm_errors": self.llm_errors,
            "judge_checks": self.judge_checks,
            "judge_fails": self.judge_fails,
            "judge_fail_rate": judge_fail_rate,
            "alerts": [
                {
                    "metric": a.metric,
                    "value": a.value,
                    "threshold": a.threshold,
                    "message": a.message,
                }
                for a in self.alerts
            ],
        }

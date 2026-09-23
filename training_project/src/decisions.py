"""Autolabel decisions: a Jev-backed decider with a threshold-based stub
fallback for when no TYPESAFE_API_KEY is configured.

Jev (docs.typesafe.ai) is text/JSON-only -- it cannot see the cropped image
itself, only structured evidence about it (see src/features.py). The stub
mimics the eventual Jev decision using YOLO confidence alone, so the rest of
the autolabeling pipeline (routing, review queue, promotion) is fully
testable without a TypeSafe account.
"""

import json
import os
from dataclasses import dataclass
from typing import Literal

Verdict = Literal["accept", "flag_for_review", "reject"]


@dataclass
class LabelDecision:
    verdict: Verdict
    score: float
    reason: str


class AutolabelDecider:
    """Decides whether an auto-detected candidate should be accepted into
    the training set, flagged for human review, or rejected."""

    def decide(self, evidence: dict) -> LabelDecision:
        raise NotImplementedError


class ThresholdStubDecider(AutolabelDecider):
    """Fallback decider used when TYPESAFE_API_KEY isn't set."""

    def __init__(self, accept_threshold: float = 0.75, reject_threshold: float = 0.35):
        self.accept_threshold = accept_threshold
        self.reject_threshold = reject_threshold

    def decide(self, evidence: dict) -> LabelDecision:
        conf = evidence.get("yolo_confidence", 0.0)
        if conf >= self.accept_threshold:
            return LabelDecision("accept", conf, "stub: yolo confidence >= accept threshold")
        if conf < self.reject_threshold:
            return LabelDecision("reject", conf, "stub: yolo confidence < reject threshold")
        return LabelDecision("flag_for_review", conf, "stub: yolo confidence in the uncertain band")


class JevDecider(AutolabelDecider):
    """Real decider backed by TypeSafe AI's Jev model."""

    def __init__(self, api_key: str | None = None):
        from typesafe_sdk import Choice, TypeSafeClient

        self._client = TypeSafeClient(api_key=api_key)
        self._Choice = Choice

    def decide(self, evidence: dict) -> LabelDecision:
        response = self._client.system_one(
            state=json.dumps(evidence),
            questions={
                "verdict": self._Choice(
                    instructions=(
                        "Given this evidence about a candidate region an object "
                        "detector flagged in an image, should it be accepted into "
                        "the training set, flagged for human review, or rejected?"
                    ),
                    criteria={
                        "accept": "Strong evidence this is a genuine match for the label",
                        "flag_for_review": "Ambiguous or conflicting evidence",
                        "reject": "Strong evidence this is not a genuine match",
                    },
                ),
            },
        )
        answer = response.answers["verdict"]
        return LabelDecision(answer.choice, answer.confidence, "jev")


def get_decider(config=None) -> AutolabelDecider:
    """Returns a real Jev-backed decider if TYPESAFE_API_KEY is set,
    otherwise a threshold-based stub."""
    api_key = os.environ.get("TYPESAFE_API_KEY")
    if api_key:
        return JevDecider(api_key=api_key)

    kwargs = {}
    if config is not None:
        if hasattr(config, "AUTOLABEL_ACCEPT_THRESHOLD"):
            kwargs["accept_threshold"] = config.AUTOLABEL_ACCEPT_THRESHOLD
        if hasattr(config, "AUTOLABEL_REJECT_THRESHOLD"):
            kwargs["reject_threshold"] = config.AUTOLABEL_REJECT_THRESHOLD
    return ThresholdStubDecider(**kwargs)

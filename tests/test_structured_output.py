"""Unit tests for `backend.agents.structured_output.invoke_structured`
(the shared None-retry / ValidationError-retry helper -- see that module's
docstring for the two free-tier failure modes it exists to paper over).

Pure Pydantic/fake-Runnable tests -- no I/O, no LLM, no skip conditions
needed, matching `tests/test_agent_state.py`'s convention.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from backend.agents.structured_output import invoke_structured


class _ExpectedResult(BaseModel):
    value: int


class _Throwaway(BaseModel):
    """Used only to naturally trigger a real pydantic_core `ValidationError`
    (via `ValidationError.from_exception_data` would require hand-building
    `InitErrorDetails` for a synthetic error type; actually invoking
    validation against a throwaway model produces a real one with zero
    guesswork about pydantic v2's internal constructor shape)."""

    x: int


def _real_validation_error() -> ValidationError:
    try:
        _Throwaway(x="not-an-int")
    except ValidationError as exc:
        return exc
    raise AssertionError("expected pydantic to raise ValidationError")


class _FakeRunnable:
    """Fake `Runnable` whose `.invoke()` replays a fixed sequence of
    results/exceptions, one per call, matching the shape
    `invoke_structured` actually calls (`.invoke(messages)`)."""

    def __init__(self, outcomes: list):
        self._outcomes = list(outcomes)
        self.call_count = 0

    def invoke(self, messages):  # noqa: ARG002
        self.call_count += 1
        outcome = self._outcomes[self.call_count - 1]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def test_retries_then_succeeds_after_validation_errors():
    """First N-1 calls raise ValidationError, the last call within budget
    returns a valid instance -- the valid result must win."""
    outcomes = [_real_validation_error(), _real_validation_error(), _ExpectedResult(value=42)]
    fake = _FakeRunnable(outcomes)

    result = invoke_structured(fake, messages=[], expected_type=_ExpectedResult, max_attempts=3)

    assert result == _ExpectedResult(value=42)
    assert fake.call_count == 3


def test_validation_error_propagates_when_every_attempt_fails():
    """Every attempt raises ValidationError -- the real ValidationError must
    propagate as-is (not masked as TypeError), since it carries the real
    "which field, why" detail."""
    outcomes = [_real_validation_error(), _real_validation_error(), _real_validation_error()]
    fake = _FakeRunnable(outcomes)

    with pytest.raises(ValidationError):
        invoke_structured(fake, messages=[], expected_type=_ExpectedResult, max_attempts=3)

    assert fake.call_count == 3


def test_type_error_when_every_attempt_returns_none():
    """A free-tier model that ignores the forced tool_choice and replies
    with plain text -- LangChain's parser turns that into `None`, no
    exception. Still `None` after every attempt must raise TypeError, same
    message shape the call sites used to raise inline."""
    fake = _FakeRunnable([None, None, None])

    with pytest.raises(TypeError, match="expected _ExpectedResult"):
        invoke_structured(fake, messages=[], expected_type=_ExpectedResult, max_attempts=3)

    assert fake.call_count == 3


def test_retries_then_succeeds_after_none():
    """First call returns None, second call returns a valid instance --
    the None case must also retry-then-succeed, not just the
    ValidationError case."""
    fake = _FakeRunnable([None, _ExpectedResult(value=7)])

    result = invoke_structured(fake, messages=[], expected_type=_ExpectedResult, max_attempts=3)

    assert result == _ExpectedResult(value=7)
    assert fake.call_count == 2

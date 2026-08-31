"""Shared helper for structured-output LLM calls that can silently return `None`
or raise on malformed tool-call JSON.

`with_structured_output(...).with_retry(retry_if_exception_type=TRANSIENT_OPENROUTER_ERRORS)`
only retries when the underlying Runnable *raises*. A free-tier OpenRouter model
can fail in two distinct ways that slip past that binding:
  1. It ignores the forced `tool_choice` and replies with plain text instead of
     calling the schema tool. LangChain's `PydanticToolsParser(first_tool_only=True)`
     sees an empty `tool_calls` list and just returns `None` -- no exception.
  2. It calls the tool but emits incomplete/invalid JSON args (e.g. a required
     nested field missing several items deep in a list). `PydanticToolsParser.
     parse_result` does `tool(**args)`, which raises `pydantic_core.ValidationError`
     straight out of `structured_llm.invoke(...)` -- not one of the transient
     OpenRouter response errors below, so `.with_retry(...)` doesn't catch it either.
Every structured-output call site needs to retry both cases -- this is the one
place that logic lives so five call sites don't each reimplement it slightly
differently.
"""

from __future__ import annotations

from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from openrouter.errors import (
    BadGatewayResponseError,
    EdgeNetworkTimeoutResponseError,
    InternalServerResponseError,
    NoResponseError,
    ProviderOverloadedResponseError,
    RequestTimeoutResponseError,
    ServiceUnavailableResponseError,
    TooManyRequestsResponseError,
)
from pydantic import ValidationError

# The `OpenRouterError` hierarchy also includes permanent, never-retryable
# errors -- UnauthorizedResponseError (401, bad API key), PaymentRequiredResponseError
# (402, out of credits), BadRequestResponseError (400, malformed/oversized prompt),
# ForbiddenResponseError (403), NotFoundResponseError (404), PayloadTooLargeResponseError
# (413), ConflictResponseError, UnprocessableEntityResponseError. Retrying those burns
# 5 attempts of exponential backoff before surfacing an error that was never going to
# succeed (e.g. a bad API key "hangs" for ~15s instead of failing instantly). This tuple
# is deliberately narrowed to the subset that's actually transient, shared by every
# `.with_retry(retry_if_exception_type=...)` call site (investigation_node, investigator,
# response_planner_node, root_cause_node, triage_node) instead of each repeating it.
# `NoResponseError` (a genuinely dropped connection) does NOT inherit from
# `OpenRouterError` but is the most transient failure of all, so it's included here too.
TRANSIENT_OPENROUTER_ERRORS = (
    ProviderOverloadedResponseError,
    TooManyRequestsResponseError,
    ServiceUnavailableResponseError,
    InternalServerResponseError,
    BadGatewayResponseError,
    RequestTimeoutResponseError,
    EdgeNetworkTimeoutResponseError,
    NoResponseError,
)


def invoke_structured[ResultT](
    structured_llm: Runnable,
    messages: list[BaseMessage],
    expected_type: type[ResultT],
    *,
    max_attempts: int = 3,
) -> ResultT:
    """Invoke `structured_llm`, retrying up to `max_attempts` times total across
    both failure modes above, then raise.

    A `ValidationError` on the last attempt is re-raised as-is (it already
    carries the real "which field, why" detail -- masking it as `TypeError`
    would hide that from logs/tests). Only the "still None after retries" case
    raises `TypeError`, same message shape the call sites used to raise inline.
    """
    result = None
    last_error: ValidationError | None = None
    for _ in range(max_attempts):
        try:
            result = structured_llm.invoke(messages)
        except ValidationError as exc:
            last_error = exc
            continue
        if isinstance(result, expected_type):
            return result
        last_error = None

    if last_error is not None:
        raise last_error

    raise TypeError(
        f"expected {expected_type.__name__} from structured output, got {type(result)!r}"
    )

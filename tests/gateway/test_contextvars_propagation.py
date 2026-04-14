"""Tests that gateway agent threads propagate contextvars.

asyncio.to_thread() propagates contextvars automatically, whereas
loop.run_in_executor() does not.  Issue #9354 showed that session
context (platform, chat_id, thread_id) was lost when the agent ran
in a thread pool, breaking tools that call get_session_env().

These tests verify that contextvars survive into worker threads when
using asyncio.to_thread().
"""

import asyncio
import contextvars

import pytest


# A minimal contextvar to simulate gateway session context
_TEST_VAR: contextvars.ContextVar[str] = contextvars.ContextVar("_TEST_VAR", default="")


class TestContextVarsPropagation:
    """Verify contextvars propagation behaviour of asyncio.to_thread vs run_in_executor."""

    @pytest.mark.asyncio
    async def test_to_thread_propagates_contextvars(self):
        """asyncio.to_thread should propagate contextvars to the worker thread."""
        _TEST_VAR.set("hello-from-gateway")

        def worker():
            return _TEST_VAR.get()

        result = await asyncio.to_thread(worker)
        assert result == "hello-from-gateway"

    @pytest.mark.asyncio
    async def test_run_in_executor_does_not_propagate(self):
        """loop.run_in_executor does NOT propagate contextvars (the bug)."""
        _TEST_VAR.set("should-not-appear")

        def worker():
            return _TEST_VAR.get()

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, worker)
        # run_in_executor does NOT propagate; worker sees the default
        assert result == ""

    @pytest.mark.asyncio
    async def test_ensure_future_to_thread_propagates(self):
        """asyncio.ensure_future(asyncio.to_thread(...)) should also propagate."""
        _TEST_VAR.set("timeout-path")

        def worker():
            return _TEST_VAR.get()

        task = asyncio.ensure_future(asyncio.to_thread(worker))
        result = await task
        assert result == "timeout-path"

    @pytest.mark.asyncio
    async def test_gateway_session_context_pattern(self):
        """Simulate the gateway session_context pattern with multiple vars."""
        platform_var: contextvars.ContextVar[str] = contextvars.ContextVar("platform", default="")
        chat_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("chat_id", default="")
        thread_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("thread_id", default="")

        platform_var.set("telegram")
        chat_id_var.set("12345")
        thread_id_var.set("topic-678")

        def run_sync():
            return {
                "platform": platform_var.get(),
                "chat_id": chat_id_var.get(),
                "thread_id": thread_id_var.get(),
            }

        result = await asyncio.to_thread(run_sync)
        assert result["platform"] == "telegram"
        assert result["chat_id"] == "12345"
        assert result["thread_id"] == "topic-678"

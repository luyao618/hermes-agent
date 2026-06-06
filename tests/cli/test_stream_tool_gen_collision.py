"""Regression test for #40693.

When streaming text accumulates in `_stream_buf` without newline boundaries
and the model transitions to a tool call, `_on_tool_gen_start` must:
  1. Emit the buffered streaming text via `_cprint` (through `_flush_stream`)
  2. Emit the response-box bottom border
  3. Then emit the "preparing <tool>…" line — AFTER the streamed text.

On Windows Terminal + Git Bash, the back-to-back `_cprint` calls race in
prompt_toolkit's StdoutProxy and the tool-progress line visually displaces
the streamed text. The fix mirrors the turn-end flush pattern (sys.stdout
.flush() + brief sleep) between the box-close and the "preparing" line.
"""
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def _make_cli_stub():
    from cli import HermesCLI

    cli = HermesCLI.__new__(HermesCLI)
    cli.show_reasoning = False
    cli.show_timestamps = False
    cli.final_response_markdown = None
    cli._stream_buf = ""
    cli._stream_started = False
    cli._stream_box_opened = False
    cli._stream_prefilt = ""
    cli._in_reasoning_block = False
    cli._reasoning_stream_started = False
    cli._reasoning_box_opened = False
    cli._reasoning_buf = ""
    cli._reasoning_preview_buf = ""
    cli._deferred_content = ""
    cli._stream_text_ansi = ""
    cli._stream_needs_break = False
    cli._stream_table_buf = []
    cli._in_stream_table = False
    cli._stream_last_was_newline = True
    return cli


def test_buffered_stream_emitted_before_tool_progress_line():
    """Streamed text without a trailing newline must be emitted before the
    'preparing <tool>…' line, with the box-bottom border in between."""
    cli = _make_cli_stub()

    captured = []

    def fake_cprint(text):
        captured.append(text)

    # Avoid sleeping in tests
    with patch("cli._cprint", side_effect=fake_cprint), \
         patch("cli.time.sleep"), \
         patch("cli.sys.stdout.flush"):
        # Drive deltas with no newline — text accumulates in _stream_buf
        cli._stream_delta("Hello, let me ")
        cli._stream_delta("check the file")
        # Tool-call begins before any '\n' arrives
        cli._on_tool_gen_start("read_file")

    joined = "\n".join(captured)

    # The streamed text should appear in the output
    assert "Hello, let me check the file" in joined, (
        f"Streamed text missing from emitted output. Got:\n{joined}"
    )

    # Locate the relevant emissions
    streamed_idx = next(
        (i for i, t in enumerate(captured) if "Hello, let me check the file" in t),
        None,
    )
    bottom_idx = next(
        (i for i, t in enumerate(captured) if "╰" in t),
        None,
    )
    preparing_idx = next(
        (i for i, t in enumerate(captured) if "preparing read_file" in t),
        None,
    )

    assert streamed_idx is not None, "Streamed text was never emitted"
    assert bottom_idx is not None, "Response-box bottom border was never emitted"
    assert preparing_idx is not None, "'preparing read_file…' line was never emitted"

    # Order: streamed text → box bottom → preparing line
    assert streamed_idx < bottom_idx < preparing_idx, (
        f"Emission order wrong: streamed={streamed_idx}, "
        f"bottom={bottom_idx}, preparing={preparing_idx}\n"
        f"Captured:\n{joined}"
    )


def test_flush_and_sleep_invoked_between_box_close_and_tool_line():
    """The renderer drain (sys.stdout.flush + time.sleep) must run between
    `_flush_stream` and the `preparing` `_cprint`, otherwise the tool-progress
    line races the streamed text in the StdoutProxy queue."""
    cli = _make_cli_stub()
    call_order = []

    def fake_cprint(text):
        call_order.append(("cprint", text))

    def fake_flush():
        call_order.append(("flush", None))

    def fake_sleep(_):
        call_order.append(("sleep", None))

    with patch("cli._cprint", side_effect=fake_cprint), \
         patch("cli.sys.stdout.flush", side_effect=fake_flush), \
         patch("cli.time.sleep", side_effect=fake_sleep):
        cli._stream_delta("buffered text")
        cli._on_tool_gen_start("read_file")

    # Find indices
    last_box_emit = max(
        (i for i, (k, t) in enumerate(call_order) if k == "cprint" and "╰" in (t or "")),
        default=-1,
    )
    flush_idx = next(
        (i for i, (k, _) in enumerate(call_order) if k == "flush"),
        -1,
    )
    sleep_idx = next(
        (i for i, (k, _) in enumerate(call_order) if k == "sleep"),
        -1,
    )
    preparing_idx = next(
        (i for i, (k, t) in enumerate(call_order) if k == "cprint" and "preparing" in (t or "")),
        -1,
    )

    assert last_box_emit >= 0, "Box bottom never emitted"
    assert flush_idx > last_box_emit, "stdout.flush() must come after box-close"
    assert sleep_idx > flush_idx, "time.sleep must come after stdout.flush"
    assert preparing_idx > sleep_idx, "preparing line must come after the drain"


def test_no_drain_when_no_open_box():
    """If no streaming box was open, we shouldn't spuriously flush+sleep."""
    cli = _make_cli_stub()
    flush_calls = []
    sleep_calls = []

    with patch("cli._cprint"), \
         patch("cli.sys.stdout.flush", side_effect=lambda: flush_calls.append(1)), \
         patch("cli.time.sleep", side_effect=lambda _: sleep_calls.append(1)):
        cli._on_tool_gen_start("read_file")

    assert flush_calls == [], "Should not flush when no box was open"
    assert sleep_calls == [], "Should not sleep when no box was open"

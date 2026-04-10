"""
Tests for Slack image/audio download content validation.

Covers: SlackAdapter._download_slack_file content-type and magic-byte
        validation added to prevent caching HTML sign-in pages as media.

See: https://github.com/NousResearch/hermes-agent/issues/6829
"""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx

from gateway.config import Platform, PlatformConfig

# ---------------------------------------------------------------------------
# Mock the slack-bolt package if it's not installed
# ---------------------------------------------------------------------------


def _ensure_slack_mock():
    """Install mock slack modules so SlackAdapter can be imported."""
    if "slack_bolt" in sys.modules and hasattr(sys.modules["slack_bolt"], "__file__"):
        return

    slack_bolt = MagicMock()
    slack_bolt.async_app.AsyncApp = MagicMock
    slack_bolt.adapter.socket_mode.async_handler.AsyncSocketModeHandler = MagicMock

    slack_sdk = MagicMock()
    slack_sdk.web.async_client.AsyncWebClient = MagicMock

    for name, mod in [
        ("slack_bolt", slack_bolt),
        ("slack_bolt.async_app", slack_bolt.async_app),
        ("slack_bolt.adapter", slack_bolt.adapter),
        ("slack_bolt.adapter.socket_mode", slack_bolt.adapter.socket_mode),
        (
            "slack_bolt.adapter.socket_mode.async_handler",
            slack_bolt.adapter.socket_mode.async_handler,
        ),
        ("slack_sdk", slack_sdk),
        ("slack_sdk.web", slack_sdk.web),
        ("slack_sdk.web.async_client", slack_sdk.web.async_client),
    ]:
        sys.modules.setdefault(name, mod)


_ensure_slack_mock()

import gateway.platforms.slack as _slack_mod

_slack_mod.SLACK_AVAILABLE = True

from gateway.platforms.slack import SlackAdapter  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def adapter():
    config = PlatformConfig(enabled=True, token="xoxb-fake-token")
    a = SlackAdapter(config)
    a._app = MagicMock()
    a._app.client = AsyncMock()
    a._bot_user_id = "U_BOT"
    a._running = True
    a.handle_message = AsyncMock()
    return a


@pytest.fixture(autouse=True)
def _redirect_caches(tmp_path, monkeypatch):
    """Point image and audio caches to tmp_path so tests don't touch ~/.hermes."""
    monkeypatch.setattr(
        "gateway.platforms.base.IMAGE_CACHE_DIR", tmp_path / "img_cache"
    )
    monkeypatch.setattr(
        "gateway.platforms.base.AUDIO_CACHE_DIR", tmp_path / "audio_cache"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Minimal valid media payloads (just enough for magic-byte detection).
FAKE_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
FAKE_JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 32
FAKE_GIF = b"GIF89a" + b"\x00" * 32
FAKE_WEBP = b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 32
FAKE_OGG = b"OggS" + b"\x00" * 32
FAKE_MP3_ID3 = b"ID3" + b"\x00" * 32
FAKE_MP3_SYNC = bytes([0xFF, 0xFB, 0x90, 0x00]) + b"\x00" * 32

# A typical HTML sign-in page payload that Slack returns on auth failure.
HTML_SIGN_IN = (
    b"<!DOCTYPE html><html><head><title>Slack</title></head>"
    b"<body>You need to sign in to see this page.</body></html>"
)


def _make_response(content: bytes, content_type: str = "image/png", status_code: int = 200):
    """Build a mock httpx.Response with the given content and headers."""
    resp = MagicMock()
    resp.content = content
    resp.status_code = status_code
    resp.headers = {"content-type": content_type}
    resp.raise_for_status = MagicMock()
    return resp


def _make_client(response):
    """Wrap a mock response in an AsyncClient context manager."""
    client = AsyncMock()
    client.get = AsyncMock(return_value=response)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


# ---------------------------------------------------------------------------
# _looks_like_media – unit tests for the static helper
# ---------------------------------------------------------------------------


class TestLooksLikeMedia:
    """Unit tests for SlackAdapter._looks_like_media."""

    def test_png_recognised(self):
        assert SlackAdapter._looks_like_media(FAKE_PNG) is True

    def test_jpeg_recognised(self):
        assert SlackAdapter._looks_like_media(FAKE_JPEG) is True

    def test_gif_recognised(self):
        assert SlackAdapter._looks_like_media(FAKE_GIF) is True

    def test_webp_recognised(self):
        assert SlackAdapter._looks_like_media(FAKE_WEBP) is True

    def test_ogg_recognised_as_audio(self):
        assert SlackAdapter._looks_like_media(FAKE_OGG, audio=True) is True

    def test_mp3_id3_recognised_as_audio(self):
        assert SlackAdapter._looks_like_media(FAKE_MP3_ID3, audio=True) is True

    def test_mp3_sync_recognised_as_audio(self):
        assert SlackAdapter._looks_like_media(FAKE_MP3_SYNC, audio=True) is True

    def test_html_rejected_as_image(self):
        assert SlackAdapter._looks_like_media(HTML_SIGN_IN) is False

    def test_html_rejected_as_audio(self):
        assert SlackAdapter._looks_like_media(HTML_SIGN_IN, audio=True) is False

    def test_empty_bytes_rejected(self):
        assert SlackAdapter._looks_like_media(b"") is False

    def test_short_bytes_rejected(self):
        assert SlackAdapter._looks_like_media(b"AB") is False

    def test_random_bytes_rejected(self):
        assert SlackAdapter._looks_like_media(b"\x00\x01\x02\x03\x04\x05") is False

    def test_ogg_not_recognised_as_image(self):
        """OGG magic bytes should not pass image validation."""
        assert SlackAdapter._looks_like_media(FAKE_OGG, audio=False) is False


# ---------------------------------------------------------------------------
# _content_type_is_html – unit tests
# ---------------------------------------------------------------------------


class TestContentTypeIsHtml:
    def test_text_html(self):
        assert SlackAdapter._content_type_is_html({"content-type": "text/html"}) is True

    def test_text_html_charset(self):
        assert SlackAdapter._content_type_is_html(
            {"content-type": "text/html; charset=utf-8"}
        ) is True

    def test_text_xml(self):
        assert SlackAdapter._content_type_is_html({"content-type": "text/xml"}) is True

    def test_image_png(self):
        assert SlackAdapter._content_type_is_html({"content-type": "image/png"}) is False

    def test_missing_header(self):
        assert SlackAdapter._content_type_is_html({}) is False


# ---------------------------------------------------------------------------
# _download_slack_file – integration tests with mocked HTTP
# ---------------------------------------------------------------------------


class TestDownloadSlackFileValidation:
    """Verify that _download_slack_file rejects non-media payloads."""

    def test_valid_image_cached_successfully(self, adapter, tmp_path):
        """A genuine PNG payload should be cached without error."""
        response = _make_response(FAKE_PNG, content_type="image/png")
        client = _make_client(response)

        async def run():
            with patch("httpx.AsyncClient", return_value=client):
                return await adapter._download_slack_file(
                    "https://files.slack.com/img.png", ".png"
                )

        path = asyncio.run(run())
        assert path.endswith(".png")
        assert os.path.exists(path)
        # Verify the cached content matches
        with open(path, "rb") as f:
            assert f.read() == FAKE_PNG

    def test_valid_audio_cached_successfully(self, adapter, tmp_path):
        """A genuine OGG payload should be cached when audio=True."""
        response = _make_response(FAKE_OGG, content_type="audio/ogg")
        client = _make_client(response)

        async def run():
            with patch("httpx.AsyncClient", return_value=client):
                return await adapter._download_slack_file(
                    "https://files.slack.com/voice.ogg", ".ogg", audio=True
                )

        path = asyncio.run(run())
        assert path.endswith(".ogg")
        assert os.path.exists(path)

    def test_html_content_type_rejected(self, adapter):
        """An HTML Content-Type should raise ValueError before caching."""
        response = _make_response(HTML_SIGN_IN, content_type="text/html; charset=utf-8")
        client = _make_client(response)

        async def run():
            with patch("httpx.AsyncClient", return_value=client):
                await adapter._download_slack_file(
                    "https://files.slack.com/img.png", ".png"
                )

        with pytest.raises(ValueError, match="HTML response"):
            asyncio.run(run())

    def test_html_body_with_image_content_type_rejected(self, adapter):
        """HTML body masquerading with image/png Content-Type should be
        rejected by magic-byte validation."""
        response = _make_response(HTML_SIGN_IN, content_type="image/png")
        client = _make_client(response)

        async def run():
            with patch("httpx.AsyncClient", return_value=client):
                await adapter._download_slack_file(
                    "https://files.slack.com/img.png", ".png"
                )

        with pytest.raises(ValueError, match="does not match any known"):
            asyncio.run(run())

    def test_html_audio_content_type_rejected(self, adapter):
        """An HTML Content-Type should be rejected for audio downloads too."""
        response = _make_response(HTML_SIGN_IN, content_type="text/html")
        client = _make_client(response)

        async def run():
            with patch("httpx.AsyncClient", return_value=client):
                await adapter._download_slack_file(
                    "https://files.slack.com/voice.ogg", ".ogg", audio=True
                )

        with pytest.raises(ValueError, match="HTML response"):
            asyncio.run(run())

    def test_garbage_bytes_rejected(self, adapter):
        """Random non-media bytes with a valid Content-Type should still be
        rejected by the magic-byte check."""
        garbage = b"\x00\x01\x02\x03garbage data that is not an image"
        response = _make_response(garbage, content_type="application/octet-stream")
        client = _make_client(response)

        async def run():
            with patch("httpx.AsyncClient", return_value=client):
                await adapter._download_slack_file(
                    "https://files.slack.com/img.png", ".png"
                )

        with pytest.raises(ValueError, match="does not match"):
            asyncio.run(run())

    def test_jpeg_with_octet_stream_content_type_accepted(self, adapter, tmp_path):
        """A valid JPEG payload should be accepted even if the Content-Type
        is generic (Slack sometimes returns application/octet-stream)."""
        response = _make_response(FAKE_JPEG, content_type="application/octet-stream")
        client = _make_client(response)

        async def run():
            with patch("httpx.AsyncClient", return_value=client):
                return await adapter._download_slack_file(
                    "https://files.slack.com/photo.jpg", ".jpg"
                )

        path = asyncio.run(run())
        assert path.endswith(".jpg")
        assert os.path.exists(path)


# ---------------------------------------------------------------------------
# End-to-end: _handle_slack_message with image validation
# ---------------------------------------------------------------------------


class TestSlackImageAttachmentValidation:
    """Verify that HTML-as-image is gracefully skipped in message handling."""

    def _make_event(self, files=None, text="hello"):
        return {
            "text": text,
            "user": "U_USER",
            "channel": "C123",
            "channel_type": "im",
            "ts": "1234567890.000001",
            "files": files or [],
        }

    @pytest.mark.asyncio
    async def test_html_image_skipped_gracefully(self, adapter):
        """When _download_slack_file raises ValueError (invalid media),
        the handler should still be called with no media attached."""
        with patch.object(
            adapter,
            "_download_slack_file",
            new_callable=AsyncMock,
            side_effect=ValueError("HTML response"),
        ):
            event = self._make_event(
                files=[
                    {
                        "mimetype": "image/png",
                        "name": "screenshot.png",
                        "url_private_download": "https://files.slack.com/screenshot.png",
                        "size": 1024,
                    }
                ]
            )
            await adapter._handle_slack_message(event)

        # Handler should still be called (error is caught)
        adapter.handle_message.assert_called_once()
        msg_event = adapter.handle_message.call_args[0][0]
        # No media should be attached since the download was invalid
        assert len(msg_event.media_urls) == 0

    @pytest.mark.asyncio
    async def test_valid_image_still_works(self, adapter):
        """A valid image download should still produce a PHOTO message."""
        with patch.object(
            adapter,
            "_download_slack_file",
            new_callable=AsyncMock,
            return_value="/tmp/cached_valid.png",
        ):
            event = self._make_event(
                files=[
                    {
                        "mimetype": "image/jpeg",
                        "name": "photo.jpg",
                        "url_private_download": "https://files.slack.com/photo.jpg",
                        "size": 2048,
                    }
                ]
            )
            await adapter._handle_slack_message(event)

        from gateway.platforms.base import MessageType

        msg_event = adapter.handle_message.call_args[0][0]
        assert msg_event.message_type == MessageType.PHOTO
        assert len(msg_event.media_urls) == 1

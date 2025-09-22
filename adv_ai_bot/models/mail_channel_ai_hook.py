# adv_ai_bot/models/mail_channel_ai_hook.py
# -*- coding: utf-8 -*-
import os
import logging
from typing import Optional, Dict, Any

import httpx
from odoo import models, tools

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHAT_MODEL = os.environ.get("ADV_AI_CHAT_MODEL", "deepseek/deepseek-r1")
try:
    TEMPERATURE = float(os.environ.get("ADV_AI_TEMPERATURE", "0.2"))
except Exception:
    TEMPERATURE = 0.2

REQUEST_TIMEOUT = float(os.environ.get("ADV_AI_TIMEOUT", "20.0"))

OPENROUTER_SITE_URL = os.environ.get("OPENROUTER_SITE_URL", "http://localhost")
OPENROUTER_APP_NAME = os.environ.get("OPENROUTER_APP_NAME", "Odoo Adv AI Bot")
USER_AGENT = os.environ.get("ADV_AI_USER_AGENT", "adv_ai_bot/1.0 (+odoo) httpx")

# If not set, we default to OpenRouter:
BASE_URL = (os.environ.get("OPENAI_BASE_URL") or "").strip() or "https://openrouter.ai/api/v1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mask_key(key: Optional[str]) -> str:
    if not key:
        return "<empty>"
    k = key
    if len(k) <= 10:
        return k[:2] + "…"
    return f"{k[:6]}…{k[-4:]}"


def _normalize_key(raw: Optional[str]) -> Optional[str]:
    """Trim, remove quotes, drop accidental 'Bearer ' prefix, keep first non-empty line."""
    if not raw:
        return None
    key = raw.strip()
    if "\n" in key:
        for line in key.splitlines():
            line = line.strip()
            if line:
                key = line
                break
    if (key.startswith('"') and key.endswith('"')) or (key.startswith("'") and key.endswith("'")):
        key = key[1:-1].strip()
    if key.lower().startswith("bearer "):
        key = key[7:].strip()
    return key or None


def _clean_network_env() -> None:
    """Avoid proxy interference which can strip Authorization headers."""
    for var in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY"):
        if os.environ.get(var):
            _logger.info("adv_ai_bot: Unsetting %s to avoid proxy interference", var)
            os.environ.pop(var, None)
    # Keep NO_PROXY if the user set it.


def _openrouter_headers(key: str) -> Dict[str, str]:
    """Headers matching your successful PowerShell request."""
    return {
        "Authorization": f"Bearer {key}",
        "openrouter-api-key": key,
        "Referer": OPENROUTER_SITE_URL,        # standard referer
        "HTTP-Referer": OPENROUTER_SITE_URL,   # OpenRouter docs variant
        "X-Title": OPENROUTER_APP_NAME,
        "User-Agent": USER_AGENT,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _post_with_redirects(url: str, headers: Dict[str, str], json: Dict[str, Any]) -> httpx.Response:
    """
    POST and, if redirected, replay the POST to the Location while preserving ALL headers.
    We keep HTTP/2 off here for maximum compatibility in strict networks.
    """
    with httpx.Client(timeout=REQUEST_TIMEOUT, trust_env=False, http2=False, follow_redirects=False) as hx:
        resp = hx.post(url, headers=headers, json=json)
        if resp.status_code in (301, 302, 303, 307, 308):
            loc = resp.headers.get("Location")
            _logger.info("adv_ai_bot: redirect %s -> %s", resp.status_code, loc)
            if not loc:
                return resp
            if loc.startswith("/"):
                parsed = httpx.URL(url)
                loc = str(parsed.copy_with(path=loc))
            return hx.post(loc, headers=headers, json=json)
        return resp


# ---------------------------------------------------------------------------
# Core LLM call (OpenRouter only)
# ---------------------------------------------------------------------------

def _call_llm(user_text: str) -> str:
    """
    Call OpenRouter chat.completions. Returns assistant text or raises a user-facing error.
    """
    _clean_network_env()

    key = _normalize_key(os.environ.get("OPENROUTER_API_KEY"))
    if not key:
        raise RuntimeError("No OPENROUTER_API_KEY found in the environment.")

    base_url = BASE_URL.rstrip("/")
    url = f"{base_url}/chat/completions"

    headers = _openrouter_headers(key)
    payload = {
        "model": CHAT_MODEL,
        "temperature": TEMPERATURE,
        "messages": [
            {"role": "system", "content": "You are an assistant inside Odoo Discuss. Keep answers concise (<=120 words)."},
            {"role": "user", "content": user_text},
        ],
    }

    _logger.info("adv_ai_bot: POST %s (model=%s, key=%s)", url, CHAT_MODEL, _mask_key(key))

    try:
        resp = _post_with_redirects(url, headers, payload)
    except httpx.ConnectError as e:
        _logger.exception("adv_ai_bot: connect error: %s", e)
        raise RuntimeError("Cannot reach the AI backend (connect error).")
    except httpx.ReadTimeout as e:
        _logger.exception("adv_ai_bot: timeout: %s", e)
        raise RuntimeError("The AI backend timed out.")
    except Exception as e:
        _logger.exception("adv_ai_bot: unexpected httpx error: %s", e)
        raise RuntimeError("Unexpected AI error.")

    if resp.status_code == 200:
        data = resp.json()
        content = (
            (data.get("choices") or [{}])[0]
            .get("message", {})
            .get("content", "")
            or ""
        )
        return content.strip() or "I couldn't generate a reply."

    # Rich diagnostics on failure
    body_preview = (resp.text or "")[:500]
    server_msg = f"status={resp.status_code}, cf-ray={resp.headers.get('cf-ray')}, server={resp.headers.get('server')}"
    _logger.error("adv_ai_bot: LLM HTTP error: %s body=%s", server_msg, body_preview)

    if resp.status_code in (401, 403):
        dbg_headers = {k: ("<masked>" if k.lower() in ("authorization", "openrouter-api-key") else v) for k, v in headers.items()}
        _logger.error("adv_ai_bot: headers_sent=%s", dbg_headers)
        raise RuntimeError(
            "Sorry, the AI backend rejected the request (401/403).\n"
            "• Check that the API key is valid and not expired/rotated.\n"
            "• If you're on a corporate network, an SSL proxy may be stripping the Authorization header.\n"
            "  Try a clean network (VPN or hotspot) or ask IT to allowlist openrouter.ai.\n"
            "• Then try again."
        )
    if resp.status_code == 429:
        raise RuntimeError("The AI backend rate-limited the request (429). Try again later.")
    raise RuntimeError(f"AI backend error ({resp.status_code}).")


# ---------------------------------------------------------------------------
# Odoo integration
# ---------------------------------------------------------------------------

class MailChannel(models.Model):
    _inherit = "mail.channel"

    def message_post(self, **kwargs):
        res = super().message_post(**kwargs)
        for channel in self:
            try:
                channel._adv_try_bot_reply(kwargs)
            except Exception:
                _logger.exception("adv_ai_bot: auto-reply failed on channel %s", channel.id)
        return res

    def _adv_try_bot_reply(self, last_msg_kwargs):
        # Avoid loops
        if self.env.context.get("adv_ai_bot_sending"):
            return
        # DM only
        if self.channel_type != "chat":
            return

        bot_partner = self.env["res.partner"].search(
            [("name", "=", "Advanced AI Bot")], limit=1
        )
        if not bot_partner or bot_partner.id not in self.channel_partner_ids.ids:
            return

        author_partner_id = last_msg_kwargs.get("author_id") or self.env.user.partner_id.id
        if author_partner_id == bot_partner.id:
            return

        body_html = last_msg_kwargs.get("body") or ""
        user_text = tools.html2plaintext(body_html).strip()
        if not user_text:
            return

        try:
            reply_text = _call_llm(user_text)
        except Exception as e:
            reply_text = str(e) or "Sorry, I hit an AI error. Please try again."

        if reply_text:
            self.with_context(adv_ai_bot_sending=True).message_post(
                body=reply_text,
                author_id=bot_partner.id,
                message_type="comment",
                subtype_xmlid="mail.mt_comment",
            )
            _logger.info("adv_ai_bot: replied in channel id=%s", self.id)

# -*- coding: utf-8 -*-
# adv_ai_bot/chatbot.py  (or adv_ai_bot/controllers/chatbot.py)
import json
import logging
import os
from odoo import http
from odoo.http import request

import requests

_logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "deepseek/deepseek-r1"  # change if you prefer another model

def _get_openrouter_key():
    """Read API key from Odoo system parameters or environment variable."""
    icp = request.env['ir.config_parameter'].sudo()
    key = icp.get_param('adv_ai_bot.openrouter_api_key') or os.getenv('OPENROUTER_API_KEY')
    return key

def _headers(host_url: str, api_key: str):
    # OpenRouter recommends setting HTTP-Referer and X-Title
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": host_url or "http://localhost",
        "X-Title": "Odoo Agent Bot",
    }

class AdvAIBotController(http.Controller):
    @http.route('/adv_ai_bot/chat', type='json', auth='user', methods=['POST'], csrf=False)
    def chat(self, message: str = "", model: str = None, temperature: float = 0.7, max_tokens: int = 4096):
        """
        JSON request body (Odoo will pass keyword args):
        {
          "message": "Say hi and tell me a fun fact.",
          "model": "deepseek/deepseek-r1",           # optional
          "temperature": 0.7,                         # optional
          "max_tokens": 4096                          # optional
        }

        JSON response:
        { "ok": true, "text": "...", "model": "deepseek/deepseek-r1" }
        or
        { "ok": false, "error": "..." }
        """
        try:
            if not message or not isinstance(message, str):
                return {"ok": False, "error": "Missing 'message'."}

            api_key = _get_openrouter_key()
            if not api_key:
                return {
                    "ok": False,
                    "error": "OpenRouter API key not configured. "
                             "Set system parameter 'adv_ai_bot.openrouter_api_key' "
                             "or the environment variable OPENROUTER_API_KEY."
                }

            model = model or DEFAULT_MODEL
            payload = {
                "model": model,
                "messages": [
                    {"role": "user", "content": message}
                ],
                # You can include extras if you like:
                "temperature": float(temperature),
                "max_tokens": int(max_tokens),
            }

            host_url = request.httprequest.host_url
            resp = requests.post(
                OPENROUTER_URL,
                headers=_headers(host_url, api_key),
                data=json.dumps(payload),
                timeout=90,
            )
            if resp.status_code != 200:
                _logger.warning("OpenRouter error %s: %s", resp.status_code, resp.text)
                # Try to surface the API’s error, if any
                try:
                    err = resp.json().get("error", {}).get("message") or resp.text
                except Exception:
                    err = resp.text
                return {"ok": False, "error": f"OpenRouter {resp.status_code}: {err}"}

            data = resp.json()
            # Standard OpenAI-compatible shape
            content = (
                data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content")
            )
            if not content:
                # Some models may put text elsewhere; fall back gracefully
                content = data.get("choices", [{}])[0].get("text") or ""

            return {"ok": True, "text": content, "model": model}

        except requests.Timeout:
            return {"ok": False, "error": "OpenRouter request timed out."}
        except Exception as e:
            _logger.exception("Unexpected chatbot error")
            return {"ok": False, "error": str(e)}

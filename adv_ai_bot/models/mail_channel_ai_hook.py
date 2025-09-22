# adv_ai_bot/models/mail_channel_ai_hook.py
# -*- coding: utf-8 -*-
import os
import re
import json
import logging
from typing import Optional, Dict, Any, List, Tuple

import httpx
from odoo import models, tools

_logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

# Defaults per provider
DEFAULT_OPENROUTER_MODEL = os.environ.get("ADV_AI_CHAT_MODEL", "deepseek/deepseek-r1")
DEFAULT_GITHUB_MODEL = os.environ.get("ADV_AI_GITHUB_MODEL", "openai/gpt-4o-mini")

try:
    TEMPERATURE = float(os.environ.get("ADV_AI_TEMPERATURE", "0.2"))
except Exception:
    TEMPERATURE = 0.2

REQUEST_TIMEOUT = float(os.environ.get("ADV_AI_TIMEOUT", "25.0"))

OPENROUTER_SITE_URL = os.environ.get("OPENROUTER_SITE_URL", "http://localhost")
OPENROUTER_APP_NAME = os.environ.get("OPENROUTER_APP_NAME", "Odoo Adv AI Bot")
USER_AGENT = os.environ.get("ADV_AI_USER_AGENT", "adv_ai_bot/1.0 (+odoo) httpx")

# Allows overriding base via env; otherwise we infer from provider (see _resolve_base_and_provider)
BASE_URL = (os.environ.get("OPENAI_BASE_URL") or "").strip() or "https://openrouter.ai/api/v1"

# GitHub API version header (recommended by docs)
GITHUB_API_VERSION = os.environ.get("GITHUB_API_VERSION", "2022-11-28")

# If you want to let the bot read with elevated rights, set this System Parameter to "1":
#   Settings → Technical → Parameters → System Parameters:
#   Key: adv_ai_bot.use_sudo  Value: 1
def _use_sudo(env) -> bool:
    val = (env["ir.config_parameter"].sudo().get_param("adv_ai_bot.use_sudo") or "").strip()
    return val in ("1", "true", "True", "yes", "on")

# Restrict what generic calls can read
ALLOWED_MODELS = {
    "res.partner",
    "crm.lead",
    "sale.order",
    "sale.order.line",
    "account.move",       # invoices
    "account.move.line",
    "stock.picking",
    "helpdesk.ticket",
}

# =============================================================================
# Provider detection / headers
# =============================================================================

def _mask_key(key: Optional[str]) -> str:
    if not key:
        return "<empty>"
    return f"{key[:6]}…{key[-4:]}" if len(key) > 10 else key[:2] + "…"

def _normalize_key(raw: Optional[str]) -> Optional[str]:
    """Trim, remove quotes, drop accidental 'Bearer ' prefix, keep first non-empty line."""
    if not raw:
        return None
    key = raw.strip()
    if "\n" in key:
        key = next((ln.strip() for ln in key.splitlines() if ln.strip()), key)
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

def _resolve_base_and_provider(env) -> Tuple[str, str]:
    """
    Decide which provider to use and which base URL to call.
    Provider: 'github' or 'openrouter'
    Precedence:
      1) System param adv_ai_bot.provider (github|openrouter)
      2) System param adv_ai_bot.base_url
      3) Env OPENAI_BASE_URL
      4) Default to OpenRouter
    """
    icp = env["ir.config_parameter"].sudo()
    provider = (icp.get_param("adv_ai_bot.provider") or "").strip().lower()
    param_base = (icp.get_param("adv_ai_bot.base_url") or "").strip()
    base = param_base or BASE_URL

    if provider not in ("github", "openrouter"):
        if "models.github.ai" in base:
            provider = "github"
        elif "openrouter.ai" in base or not base:
            provider = "openrouter"
        else:
            provider = "openrouter"

    # Normalize base if user only gave provider
    if not param_base:
        if provider == "github":
            base = "https://models.github.ai/inference"
        else:
            base = "https://openrouter.ai/api/v1"

    return provider, base.rstrip("/")

def _get_api_key(env, provider: str) -> Optional[str]:
    """
    GitHub → PAT from system param or env; OpenRouter → same as before.
    System params:
      - adv_ai_bot.github_pat
      - adv_ai_bot.openrouter_api_key
    """
    icp = env["ir.config_parameter"].sudo()
    if provider == "github":
        key = icp.get_param("adv_ai_bot.github_pat")
        key = _normalize_key(key) or _normalize_key(os.environ.get("GITHUB_TOKEN"))
        return key
    # default: openrouter
    key = icp.get_param("adv_ai_bot.openrouter_api_key")
    key = _normalize_key(key) or _normalize_key(os.environ.get("OPENROUTER_API_KEY"))
    return key

def _headers_for_provider(provider: str, key: str) -> Dict[str, str]:
    if provider == "github":
        # https://docs.github.com/en/github-models/quickstart (headers & endpoint)
        return {
            "Authorization": f"Bearer {key}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": GITHUB_API_VERSION,
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        }
    # OpenRouter defaults
    return {
        "Authorization": f"Bearer {key}",
        "Referer": OPENROUTER_SITE_URL,
        "X-Title": OPENROUTER_APP_NAME,
        "User-Agent": USER_AGENT,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

def _post_with_redirects(url: str, headers: Dict[str, str], json_payload: Dict[str, Any]) -> httpx.Response:
    """
    POST and, if redirected, replay the POST to the Location while preserving ALL headers.
    http2=False for compatibility. trust_env=False so system proxies cannot strip headers.
    """
    with httpx.Client(timeout=REQUEST_TIMEOUT, trust_env=False, http2=False, follow_redirects=False) as hx:
        resp = hx.post(url, headers=headers, json=json_payload)
        if resp.status_code in (301, 302, 303, 307, 308):
            loc = resp.headers.get("Location")
            _logger.info("adv_ai_bot: redirect %s -> %s", resp.status_code, loc)
            if not loc:
                return resp
            if loc.startswith("/"):
                parsed = httpx.URL(url)
                loc = str(parsed.copy_with(path=loc))
            return hx.post(loc, headers=headers, json=json_payload)
        return resp

def _select_model(env, provider: str) -> str:
    icp = env["ir.config_parameter"].sudo()
    user_model = (icp.get_param("adv_ai_bot.chat_model") or "").strip()
    if user_model:
        return user_model
    return DEFAULT_GITHUB_MODEL if provider == "github" else DEFAULT_OPENROUTER_MODEL

# =============================================================================
# JSON Plan Protocol (R1-friendly)
# =============================================================================

JSON_PLAN_SYSTEM = (
    "You are an Odoo assistant. When asked about Odoo data, output ONLY a single-line JSON plan and NOTHING else.\n"
    "Schema:\n"
    '{ "call": { "name": "<one of: search_partners | search_crm_leads | search_sale_orders | odoo_search_read | odoo_count>", "args": { ... } } }\n'
    "Examples:\n"
    '{ "call": { "name": "search_partners", "args": {"query": "Deco", "limit": 5} } }\n'
    '{ "call": { "name": "search_crm_leads", "args": {"query": "azure", "stage": "Proposition", "limit": 5} } }\n'
    '{ "call": { "name": "search_sale_orders", "args": {"query": "Deco Addict", "state": "sale", "limit": 5} } }\n'
    '{ "call": { "name": "odoo_search_read", "args": {"model": "crm.lead", "domain": [["stage_id","ilike","new"]], "fields": ["name","email_from","probability"], "limit": 10, "order": "create_date desc"} } }\n'
    '{ "call": { "name": "odoo_count", "args": {"model": "sale.order", "domain": [["state","=","sale"]] } } }\n'
    'If the question is not about Odoo data, return exactly: {"call": null}\n'
)

def _extract_json_maybe(text: str) -> Optional[str]:
    """
    Try to extract a JSON object from content, tolerant to code fences or reasoning.
    """
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S | re.I)
    if fence:
        return fence.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        return text[start:end+1]
    t = text.strip()
    if t.startswith("{") and t.endswith("}"):
        return t
    return None

def _try_json_plan(env, user_text: str) -> Optional[Dict[str, Any]]:
    """
    Ask the model to give a JSON plan. Return parsed dict or None.
    """
    _clean_network_env()
    provider, base = _resolve_base_and_provider(env)
    key = _get_api_key(env, provider)
    if not key:
        return None

    url = f"{base}/chat/completions"
    headers = _headers_for_provider(provider, key)
    model = _select_model(env, provider)

    payload = {
        "model": model,
        "temperature": 0,  # strict
        "messages": [
            {"role": "system", "content": JSON_PLAN_SYSTEM},
            {"role": "user", "content": user_text},
        ],
    }

    _logger.info("adv_ai_bot: JSON plan -> %s (provider=%s, model=%s, key=%s)", url, provider, model, _mask_key(key))
    resp = _post_with_redirects(url, headers, payload)
    if resp.status_code != 200:
        _logger.error("adv_ai_bot: JSON plan error %s: %s", resp.status_code, resp.text[:400])
        return None

    try:
        data = resp.json()
        content = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        raw = _extract_json_maybe(content) or content.strip()
        plan = json.loads(raw)
        if isinstance(plan, dict) and "call" in plan:
            return plan
    except Exception:
        _logger.exception("adv_ai_bot: failed to parse JSON plan")
    return None

# =============================================================================
# Tool Implementations (server-side execution)
# =============================================================================

def _env_for_ops(env):
    return env.sudo() if _use_sudo(env) else env

def _safe_rows(rows: List[Dict[str, Any]], max_len=5) -> List[Dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    return rows[:max_len]

def _summarize(model: str, payload: Dict[str, Any]) -> str:
    if "error" in payload and payload["error"]:
        return f"Error: {payload['error']}"
    if payload.get("ok") is True and "rows" in payload:
        rows = payload.get("rows") or []
        n = len(rows)
        if n == 0:
            return "No matching records."
        head = f"Found {n} record(s) in {model}."
        lines = []
        for r in _safe_rows(rows, 5):
            if model == "res.partner":
                lines.append(f"- {r.get('name')}  <{r.get('email') or '-'}>  {r.get('phone') or r.get('mobile') or ''}".strip())
            elif model == "crm.lead":
                lines.append(f"- {r.get('name')}  (stage: {r.get('stage_id') or r.get('stage') or '-'}, prob: {r.get('probability')})")
            elif model == "sale.order":
                lines.append(f"- {r.get('name')}  {r.get('partner_id') or r.get('partner') or '-'}  {r.get('state')}  total={r.get('amount_total')}")
            else:
                lines.append("- " + ", ".join(
                    f"{k}={v}" for k, v in r.items()
                    if k in ("name", "email_from", "partner_id", "state", "amount_total", "date_order") and v is not None
                ))
        return head + ("\n" + "\n".join(lines) if lines else "")
    if payload.get("ok") is True and "count" in payload:
        return f"Count in {model}: {payload.get('count')}"
    return "I couldn’t format a reply."

def _fields_filter(model_obj, fields_req: Optional[List[str]]) -> Optional[List[str]]:
    if not fields_req:
        return None
    allowed = []
    model_fields = set(model_obj._fields.keys())
    for f in fields_req:
        if isinstance(f, str) and f in model_fields:
            allowed.append(f)
    return allowed or None

def _impl_search_partners(env, args: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    q = (args.get("query") or "").strip()
    limit = int(args.get("limit") or 5)
    if not q:
        return "res.partner", {"error": "query is required"}
    E = _env_for_ops(env)
    domain = ["|", ("name", "ilike", q), ("email", "ilike", q)]
    rows = E["res.partner"].search_read(domain, fields=["name", "email", "phone", "mobile", "parent_id"], limit=limit, order="write_date desc")
    for r in rows:
        if isinstance(r.get("parent_id"), (list, tuple)) and r["parent_id"]:
            r["company"] = r["parent_id"][1]
    return "res.partner", {"ok": True, "rows": rows}

def _impl_search_crm_leads(env, args: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    q = (args.get("query") or "").strip()
    limit = int(args.get("limit") or 5)
    stage = (args.get("stage") or "").strip()
    if not q and not stage:
        return "crm.lead", {"error": "query or stage is required"}
    E = _env_for_ops(env)
    domain = ["|", "|", ("name", "ilike", q), ("email_from", "ilike", q), ("phone", "ilike", q)] if q else []
    if stage:
        domain = ["&", ("stage_id.name", "ilike", stage)] + (domain or [("id", "!=", 0)])
    rows = E["crm.lead"].search_read(domain, fields=["name", "email_from", "phone", "stage_id", "probability", "expected_revenue", "create_date"], limit=limit, order="create_date desc")
    return "crm.lead", {"ok": True, "rows": rows}

def _impl_search_sale_orders(env, args: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    q = (args.get("query") or "").strip()
    limit = int(args.get("limit") or 5)
    state = (args.get("state") or "").strip()
    if not q and not state:
        return "sale.order", {"error": "query or state is required"}
    E = _env_for_ops(env)
    domain: List[Any] = []
    if q:
        domain = ["|", ("name", "ilike", q), "|", ("partner_id.name", "ilike", q), ("partner_id.email", "ilike", q)]
    if state:
        domain = ["&", ("state", "=", state)] + (domain or [("id", "!=", 0)])
    rows = E["sale.order"].search_read(domain, fields=["name", "state", "partner_id", "amount_total", "date_order"], limit=limit, order="date_order desc")
    return "sale.order", {"ok": True, "rows": rows}

def _impl_odoo_search_read(env, args: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    model = (args.get("model") or "").strip()
    if model not in ALLOWED_MODELS:
        return model or "unknown", {"error": f"Model '{model}' not allowed"}
    domain = args.get("domain") or []
    fields_req = args.get("fields") or None
    limit = int(args.get("limit") or 20)
    order = (args.get("order") or None)

    E = _env_for_ops(env)
    try:
        M = E[model]
    except Exception:
        return model, {"error": f"Unknown model '{model}'"}
    fields_safe = _fields_filter(M, fields_req)

    try:
        rows = M.search_read(domain, fields=fields_safe, limit=limit, order=order)
        return model, {"ok": True, "rows": rows}
    except Exception as e:
        _logger.exception("adv_ai_bot: odoo_search_read failed")
        return model, {"error": str(e) or "search_read failed"}

def _impl_odoo_count(env, args: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    model = (args.get("model") or "").strip()
    if model not in ALLOWED_MODELS:
        return model or "unknown", {"error": f"Model '{model}' not allowed"}
    domain = args.get("domain") or []
    E = _env_for_ops(env)
    try:
        cnt = E[model].search_count(domain)
        return model, {"ok": True, "count": cnt}
    except Exception as e:
        _logger.exception("adv_ai_bot: odoo_count failed")
        return model, {"error": str(e) or "count failed"}

EXEC_MAP = {
    "search_partners": _impl_search_partners,
    "search_crm_leads": _impl_search_crm_leads,
    "search_sale_orders": _impl_search_sale_orders,
    "odoo_search_read": _impl_odoo_search_read,
    "odoo_count": _impl_odoo_count,
}

# =============================================================================
# Plain LLM call (fallback chat)
# =============================================================================

def _call_llm_plain(env, user_text: str) -> str:
    _clean_network_env()
    provider, base = _resolve_base_and_provider(env)
    key = _get_api_key(env, provider)
    if not key:
        raise RuntimeError(
            "No API key configured.\n"
            "GitHub: set System Parameter 'adv_ai_bot.github_pat' or env 'GITHUB_TOKEN'.\n"
            "OpenRouter: set System Parameter 'adv_ai_bot.openrouter_api_key' or env 'OPENROUTER_API_KEY'."
        )

    url = f"{base}/chat/completions"
    model = _select_model(env, provider)
    headers = _headers_for_provider(provider, key)

    payload = {
        "model": model,
        "temperature": TEMPERATURE,
        "messages": [
            {"role": "system", "content": "You are an assistant inside Odoo Discuss. Keep answers concise (<=120 words)."},
            {"role": "user", "content": user_text},
        ],
    }

    _logger.info("adv_ai_bot: POST %s (provider=%s, model=%s, key=%s)", url, provider, model, _mask_key(key))

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
        content = ((data.get("choices") or [{}])[0].get("message", {}) or {}).get("content") or ""
        return content.strip() or "I couldn't generate a reply."

    body_preview = (resp.text or "")[:500]
    server_msg = f"status={resp.status_code}, cf-ray={resp.headers.get('cf-ray')}, server={resp.headers.get('server')}"
    _logger.error("adv_ai_bot: LLM HTTP error: %s body=%s", server_msg, body_preview)

    if resp.status_code in (401, 403):
        _logger.error("adv_ai_bot: headers_sent(auth masked)=%s", {k: ("<masked>" if k.lower()=="authorization" else v) for k, v in headers.items()})
        who = "GitHub Models" if provider == "github" else "OpenRouter"
        raise RuntimeError(
            f"Sorry, {who} rejected the request (HTTP {resp.status_code}).\n"
            "• Verify the API key/PAT is valid and has the right scope.\n"
            "• Corporate proxies sometimes strip Authorization—try a clean network or allowlist the host.\n"
            "• Then try again."
        )
    if resp.status_code == 429:
        raise RuntimeError("The AI backend rate-limited the request (429). Try again later.")
    raise RuntimeError(f"AI backend error ({resp.status_code}).")

# =============================================================================
# Orchestrator: JSON plan → execute → summarize → (fallback) plain chat
# =============================================================================

def _route_and_reply(env, user_text: str) -> str:
    """
    1) Ask model for a strict JSON plan.
    2) If present, execute mapped function and summarize.
    3) Otherwise fall back to normal chat.
    """
    plan = _try_json_plan(env, user_text)
    if plan and isinstance(plan.get("call"), dict):
        name = (plan["call"].get("name") or "").strip()
        args = plan["call"].get("args") or {}
        impl = EXEC_MAP.get(name)
        if not impl:
            return f"Tool '{name}' is not available."
        model, result = impl(env, args)
        return _summarize(model, result)

    return _call_llm_plain(env, user_text)

# =============================================================================
# Odoo integration
# =============================================================================

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

        bot_partner = self.env["res.partner"].search([("name", "=", "Advanced AI Bot")], limit=1)
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
            reply_text = _route_and_reply(self.env, user_text)
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

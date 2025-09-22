# customaddons/adv_ai_bot/hooks.py
import logging
from odoo import api, SUPERUSER_ID

_logger = logging.getLogger(__name__)

def process_initial_knowledge_base(cr, registry):
    """Post-init hook: keep it FAST and side-effect safe."""
    env = api.Environment(cr, SUPERUSER_ID, {})
    _logger.info("adv_ai_bot: post-init hook started")

    # Optional: do lightweight, safe setup here (no network calls).
    # e.g. create a default config record in your own models if you have one.
    # Avoid calling OpenAI/HTTP here; hooks run during module install/upgrade.

    _logger.info("adv_ai_bot: post-init hook complete")

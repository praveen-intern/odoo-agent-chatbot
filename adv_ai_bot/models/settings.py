# adv_ai_bot/models/settings.py
from odoo import api, fields, models

class ResConfigSettings(models.TransientModel):
    _inherit = 'res.config.settings'

    adv_openrouter_key = fields.Char(string="OpenRouter API Key", config_parameter='adv_ai_bot.openrouter_api_key')
    adv_model = fields.Char(string="Default Model", config_parameter='adv_ai_bot.default_model', default='deepseek/deepseek-r1')

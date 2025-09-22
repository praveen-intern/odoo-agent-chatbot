# -*- coding: utf-8 -*-

import logging

from odoo import models, fields

_logger = logging.getLogger(__name__)


class AdvAiBotMessage(models.Model):
    _name = 'mail.bot.advai.message'
    _description = 'Advanced AI Bot Message History'

    channel_id = fields.Many2one('mail.channel', string="Channel", ondelete='cascade')
    message = fields.Text('Message')


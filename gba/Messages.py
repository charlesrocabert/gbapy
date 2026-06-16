#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#***********************************************************************
# gbapy (growth balance analysis for Python)
# Web: https://github.com/charlesrocabert/gbapy
# Copyright © 2024-2026 Charles Rocabert.
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#***********************************************************************

"""
Filename: Messages.py
Author: Charles Rocabert
Date: 2026-06-16
Description:
    Common message utilities for the gbapy module.
License: GNU General Public License v3 (GPLv3)
Copyright: © 2024-2026 Charles Rocabert.
"""

from IPython.display import display_html

try:
    from .Enumerations import MessageType
except ImportError:
    from Enumerations import MessageType


def throw_message(type: MessageType, message: str) -> None:
    """
    Throw a message to the user.

    Parameters
    ----------
    type : MessageType
        Type of message (MessageType.INFO, MessageType.WARNING,
        MessageType.ERROR, MessageType.PLAIN).
    message : str
        Content of the message.
    """
    html_str = "<table>"
    html_str += "<tr style='text-align:left'><td style='vertical-align:top'>"
    if type == MessageType.PLAIN:
        html_str += "<td><strong>&#10095;</strong></td>"
    elif type == MessageType.INFO:
        html_str += "<td style='color:rgba(0,85,194);'><strong>&#10095; Info</strong></td>"
    elif type == MessageType.WARNING:
        html_str += "<td style='color:rgba(240,147,1);'><strong>&#9888; Warning</strong></td>"
    elif type == MessageType.ERROR:
        html_str += "<td style='color:rgba(236,3,3);'><strong>&#10006; Error</strong></td>"
    html_str += "<td>" + message + "</td>"
    html_str += "</tr>"
    html_str += "</table>"
    display_html(html_str, raw=True)


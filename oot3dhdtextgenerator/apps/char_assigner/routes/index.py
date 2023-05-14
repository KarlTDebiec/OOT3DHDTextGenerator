#  Copyright 2020-2023 Karl T Debiec. All rights reserved. This software may be modified
#  and distributed under the terms of the BSD license. See the LICENSE file for details.
from __future__ import annotations

from flask import render_template

from oot3dhdtextgenerator.apps.char_assigner.models.character import Character


def route_index(char_assigner):
    @char_assigner.app.route("/", methods=["GET"])
    def index():
        characters = []
        i = 0
        for char_bytes in char_assigner.dataset.unassigned_char_bytes:
            char_array = char_assigner.dataset.bytes_to_array(char_bytes)
            characters.append(Character(char_array, str(i)))
            i += 1

        return render_template("index.html", characters=characters)

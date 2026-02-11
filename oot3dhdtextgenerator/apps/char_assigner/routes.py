#  Copyright 2020-2026 Karl T Debiec. All rights reserved. This software may be modified
#  and distributed under the terms of the BSD license. See the LICENSE file for details.
"""Routes for the character assignment web application."""

from __future__ import annotations

from flask import render_template, request


def route(char_assigner):
    """Register routes for the Flask application."""

    @char_assigner.app.route("/", methods=["GET"])
    def index():
        """Render the character assignment index page."""
        return render_template("index.html", characters=char_assigner.characters)

    @char_assigner.app.route("/characters/<int:character_id>", methods=["PUT"])
    def update_character(character_id):
        """Update a character assignment and persist it."""
        assignment = request.form["assignment"]
        if assignment == "":
            assignment = None
        character = char_assigner.characters[character_id]
        if character.assignment != assignment:
            character.assignment = assignment
            char_assigner.dataset.assign(character.array, assignment)
            char_assigner.dataset.save_hdf5(
                char_assigner.dataset.assigned_char_bytes,
                char_assigner.dataset.unassigned_char_bytes,
                char_assigner.assignment_path,
            )

        return render_template("character.html", character=character)

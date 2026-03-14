#  Copyright 2020-2026 Karl T Debiec. All rights reserved. This software may be modified
#  and distributed under the terms of the BSD license. See the LICENSE file for details.
"""Routes for the character assignment web application."""

from __future__ import annotations

from flask import render_template, request


def route(char_assigner):
    """Register routes for the Flask application."""

    def normalize_nonnegative_int(value: str | None, default: int) -> int:
        """Normalize request integer to a nonnegative value with fallback."""
        try:
            if value is None:
                return default
            return max(0, int(value))
        except ValueError:
            return default

    @char_assigner.app.route("/", methods=["GET"])
    def index():
        """Render the character assignment index page."""
        (
            characters,
            total_count,
            has_more,
            unassigned_filter,
            assigned_filter,
            prior_weight_percent,
            exclude_assigned_from_predictions,
        ) = char_assigner.get_display_characters_page(
            request.args.get("unassigned_filter"),
            request.args.get("assigned_filter"),
            request.args.get("prior_weight_percent"),
            request.args.get("exclude_assigned_from_predictions"),
            offset=0,
            limit=char_assigner.default_page_size,
        )
        return render_template(
            "index.html",
            characters=characters,
            total_count=total_count,
            has_more=has_more,
            next_offset=len(characters),
            page_size=char_assigner.default_page_size,
            unassigned_filter=unassigned_filter,
            assigned_filter=assigned_filter,
            prior_weight_percent=prior_weight_percent,
            exclude_assigned_from_predictions=exclude_assigned_from_predictions,
        )

    @char_assigner.app.route("/characters", methods=["GET"])
    def get_characters():
        """Render filtered character rows."""
        (
            characters,
            total_count,
            has_more,
            _normalized_unassigned_filter,
            _normalized_assigned_filter,
            _normalized_prior_weight_percent,
            _normalized_exclude_assigned_from_predictions,
        ) = char_assigner.get_display_characters_page(
            request.args.get("unassigned_filter"),
            request.args.get("assigned_filter"),
            request.args.get("prior_weight_percent"),
            request.args.get("exclude_assigned_from_predictions"),
            offset=normalize_nonnegative_int(request.args.get("offset"), 0),
            limit=normalize_nonnegative_int(
                request.args.get("limit"), char_assigner.default_page_size
            ),
        )
        next_offset = normalize_nonnegative_int(request.args.get("offset"), 0) + len(
            characters
        )
        return render_template(
            "characters_rows.html",
            characters=characters,
            has_more=has_more,
            next_offset=next_offset,
            total_count=total_count,
            page_size=normalize_nonnegative_int(
                request.args.get("limit"), char_assigner.default_page_size
            ),
        )

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
            char_assigner.dataset.save_csv(
                char_assigner.dataset.assigned_char_bytes,
                char_assigner.dataset.unassigned_char_bytes,
                char_assigner.dataset.assigned_csv_path,
                char_assigner.dataset.unassigned_csv_path,
            )
            char_assigner.mark_assignments_changed()
        (
            characters,
            total_count,
            has_more,
            _normalized_unassigned_filter,
            _normalized_assigned_filter,
            _normalized_prior_weight_percent,
            _normalized_exclude_assigned_from_predictions,
        ) = char_assigner.get_display_characters_page(
            request.values.get("unassigned_filter"),
            request.values.get("assigned_filter"),
            request.values.get("prior_weight_percent"),
            request.values.get("exclude_assigned_from_predictions"),
            offset=0,
            limit=char_assigner.default_page_size,
        )
        return render_template(
            "characters_rows.html",
            characters=characters,
            has_more=has_more,
            next_offset=len(characters),
            total_count=total_count,
            page_size=char_assigner.default_page_size,
        )

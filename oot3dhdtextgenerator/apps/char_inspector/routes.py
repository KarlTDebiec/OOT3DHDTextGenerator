#  Copyright 2020-2026 Karl T Debiec. All rights reserved. This software may be modified
#  and distributed under the terms of the BSD license. See the LICENSE file for details.
"""Routes for the character inspector web application."""

from __future__ import annotations

from flask import render_template, request


def route(char_inspector):
    """Register routes for the Flask application."""

    def normalize_nonnegative_int(value: str | None, default: int) -> int:
        """Normalize request integer to a nonnegative value with fallback."""
        try:
            if value is None:
                return default
            return max(0, int(value))
        except ValueError:
            return default

    @char_inspector.app.route("/", methods=["GET"])
    def index():
        """Render character inspector index page."""
        filters = char_inspector.normalize_filters(
            request.args.getlist("font"),
            request.args.getlist("size"),
            request.args.getlist("x_offset"),
            request.args.getlist("y_offset"),
            request.args.getlist("rotation"),
            default_to_all=not bool(request.args),
        )
        (
            display_rows,
            total_count,
            has_more,
        ) = char_inspector.get_display_rows_page(
            filters,
            offset=0,
            limit=char_inspector.default_page_size,
        )
        training_headers = char_inspector.get_training_headers(filters)
        return render_template(
            "index.html",
            display_rows=display_rows,
            training_headers=training_headers,
            training_column_count=len(training_headers),
            total_columns=2 + len(training_headers),
            total_count=total_count,
            has_more=has_more,
            next_offset=len(display_rows),
            page_size=char_inspector.default_page_size,
            font_options=char_inspector.available_font_labels,
            selected_fonts=filters.fonts,
            size_options=char_inspector.available_sizes,
            selected_sizes=filters.sizes,
            offset_options=char_inspector.available_offsets,
            selected_x_offsets=filters.x_offsets,
            selected_y_offsets=filters.y_offsets,
            rotation_options=char_inspector.available_rotations,
            selected_rotations=filters.rotations,
        )

    @char_inspector.app.route("/table", methods=["GET"])
    def table():
        """Render full inspector table for filter changes."""
        filters = char_inspector.normalize_filters(
            request.args.getlist("font"),
            request.args.getlist("size"),
            request.args.getlist("x_offset"),
            request.args.getlist("y_offset"),
            request.args.getlist("rotation"),
            default_to_all=False,
        )
        (
            display_rows,
            total_count,
            has_more,
        ) = char_inspector.get_display_rows_page(
            filters,
            offset=0,
            limit=normalize_nonnegative_int(
                request.args.get("limit"), char_inspector.default_page_size
            ),
        )
        training_headers = char_inspector.get_training_headers(filters)
        return render_template(
            "table.html",
            display_rows=display_rows,
            training_headers=training_headers,
            training_column_count=len(training_headers),
            total_columns=2 + len(training_headers),
            total_count=total_count,
            has_more=has_more,
            next_offset=len(display_rows),
            page_size=normalize_nonnegative_int(
                request.args.get("limit"), char_inspector.default_page_size
            ),
        )

    @char_inspector.app.route("/rows", methods=["GET"])
    def rows():
        """Render paged character inspector rows."""
        filters = char_inspector.normalize_filters(
            request.args.getlist("font"),
            request.args.getlist("size"),
            request.args.getlist("x_offset"),
            request.args.getlist("y_offset"),
            request.args.getlist("rotation"),
            default_to_all=False,
        )
        offset = normalize_nonnegative_int(request.args.get("offset"), 0)
        limit = normalize_nonnegative_int(
            request.args.get("limit"), char_inspector.default_page_size
        )
        display_rows, total_count, has_more = char_inspector.get_display_rows_page(
            filters,
            offset=offset,
            limit=limit,
        )
        training_headers = char_inspector.get_training_headers(filters)
        return render_template(
            "rows.html",
            display_rows=display_rows,
            training_column_count=len(training_headers),
            total_count=total_count,
            has_more=has_more,
            next_offset=offset + len(display_rows),
            page_size=limit,
            total_columns=2 + len(training_headers),
        )

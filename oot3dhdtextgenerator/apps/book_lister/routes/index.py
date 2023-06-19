#  Copyright 2020-2023 Karl T Debiec. All rights reserved. This software may be modified
#  and distributed under the terms of the BSD license. See the LICENSE file for details.
from flask import render_template


def route_index(app, db):
    Author = app.Author
    Book = app.Book

    @app.route("/", methods=["GET"])
    def index():
        books_and_authors = (
            db.session.query(Book, Author)
            .filter(Book.author_id == Author.author_id)
            .all()
        )
        return render_template("index.html", books_and_authors=books_and_authors)
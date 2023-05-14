#  Copyright 2020-2023 Karl T Debiec. All rights reserved. This software may be modified
#  and distributed under the terms of the BSD license. See the LICENSE file for details.

from flask import render_template, request


def route_books(app, db):
    Author = app.Author
    Book = app.Book

    @app.route("/books", methods=["POST"])
    def create():
        book_title = request.form.get("title")
        author_name = request.form.get("author")

        author = db.session.query(Author).filter(Author.name == author_name).first()
        if not author:
            author = Author(name=author_name)
            db.session.add(author)
            db.session.commit()

        book = Book(author_id=author.author_id, title=book_title)
        db.session.add(book)
        db.session.commit()

        return render_template("books/book.html", book=book, author=author)

    @app.route("/books/<int:id>", methods=["DELETE"])
    def delete(id):
        book = Book.query.get(id)
        db.session.delete(book)
        db.session.commit()

        return ""

    @app.route("/books/<int:id>/edit", methods=["GET"])
    def edit(id):
        book = Book.query.get(id)
        author = Author.query.get(book.author_id)

        return render_template("books/edit.html", book=book, author=author)

    @app.route("/books/<int:id>", methods=["GET"])
    def read(id):
        book = Book.query.get(id)
        author = Author.query.get(book.author_id)

        return render_template("books/book.html", book=book, author=author)

    @app.route("/books/<int:id>", methods=["PUT"])
    def update(id):
        db.session.query(Book).filter(Book.book_id == id).update(
            {"title": request.form["title"]}
        )
        db.session.commit()

        book = Book.query.get(id)
        author = Author.query.get(book.author_id)

        return render_template("books/book.html", book=book, author=author)

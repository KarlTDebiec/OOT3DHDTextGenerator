#  Copyright 2020-2023 Karl T Debiec. All rights reserved. This software may be modified
#  and distributed under the terms of the BSD license. See the LICENSE file for details.

from flask import request

from oot3dhdtextgenerator.apps.char_assigner import app, db
from oot3dhdtextgenerator.apps.char_assigner.models import Author, Book


@app.route("/books", methods=["POST"])
def create():
    global_book_object = Book()

    title = request.form.get("title")
    author_name = request.form.get("author")

    author_exists = db.session.query(Author).filter(Author.name == author_name).first()
    print(author_exists)

    if author_exists:
        author_id = author_exists.author_id
        book = Book(author_id=author_id, title=title)
        db.session.add(book)
        db.session.commit()
        global_book_object = book
    else:
        author = Author(name=author_name)
        db.session.add(author)
        db.session.commit()

        book = Book(author_id=author.author_id, title=title)
        db.session.add(book)
        db.session.commit()
        global_book_object = book

    response = f"""
    <tr>
        <td>{title}</td>
        <td>{author_name}</td>
        <td>
            <button hx-get="/books/{global_book_object.book_id}/edit" class="btn btn-primary" >
                Edit Title
            </button>
        </td>
        <td>
            <button hx-delete="/books/{global_book_object.book_id}" class="btn btn-primary">
                Delete
            </button>
        </td>
    </tr>
    """
    return response


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

    response = f"""
    <tr hx-trigger='cancel' class='editing' hx-get="/get-book-row/{id}">
        <td><input name="title" value="{book.title}"/></td>
        <td>{author.name}</td>
        <td>
            <button hx-get="/books/{id}" class="btn btn-primary">
                Cancel
            </button>
            <button hx-put="/books/{id}" hx-include="closest tr" class="btn btn-primary">
                Save
            </button>
        </td>
        <td>
            <button class="btn btn-disabled">
                Delete
            </button>
        </td>
    </tr>
    """

    return response


@app.route("/books/<int:id>", methods=["GET"])
def read(id):
    book = Book.query.get(id)
    author = Author.query.get(book.author_id)

    response = f"""
    <tr>
        <td>{book.title}</td>
        <td>{author.name}</td>
        <td>
            <button hx-get="/books/{id}/edit" class="btn btn-primary">
                Edit Title
            </button>
        </td>
        <td>
            <button hx-delete="/books/{id}" class="btn btn-primary">
                Delete
            </button>
        </td>
    </tr>
    """

    return response


@app.route("/books/<int:id>", methods=["PUT"])
def update(id):
    db.session.query(Book).filter(Book.book_id == id).update(
        {"title": request.form["title"]}
    )
    db.session.commit()

    title = request.form["title"]
    book = Book.query.get(id)
    author = Author.query.get(book.author_id)

    response = f"""
    <tr>
        <td>{title}</td>
        <td>{author.name}</td>
        <td>
            <button hx-get="/books/{id}/edit" class="btn btn-primary">
                Edit Title
            </button>
        </td>
        <td>
            <button hx-delete="/books/{id}" class="btn btn-primary">
                Delete
            </button>
        </td>
    </tr>
    """

    return response

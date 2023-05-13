from app import app, db
from app.models import Author, Book
from flask import render_template, request


@app.route("/", methods=["GET"])
def home():
    books = (
        db.session.query(Book, Author).filter(Book.author_id == Author.author_id).all()
    )
    return render_template("index.html", books=books)


@app.route("/submit", methods=["POST"])
def submit():
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
            <button class="btn btn-primary"
                hx-get="/get-edit-form/{global_book_object.book_id}">
                Edit Title
            </button>
        </td>
        <td>
            <button hx-delete="/delete/{global_book_object.book_id}"
                class="btn btn-primary">
                Delete
            </button>
        </td>
    </tr>
    """
    return response


@app.route("/delete/<int:id>", methods=["DELETE"])
def delete_book(id):
    book = Book.query.get(id)
    db.session.delete(book)
    db.session.commit()

    return ""


@app.route("/get-edit-form/<int:id>", methods=["GET"])
def get_edit_form(id):
    book = Book.query.get(id)
    author = Author.query.get(book.author_id)

    response = f"""
    <tr hx-trigger='cancel' class='editing' hx-get="/get-book-row/{id}">
        <td><input name="title" value="{book.title}"/></td>
        <td>{author.name}</td>
        <td>
            <button class="btn btn-primary" hx-get="/get-book-row/{id}">
                Cancel
            </button>
            <button class="btn btn-primary" hx-put="/update/{id}" hx-include="closest tr">
                Save
            </button>
        </td>
        <td>
            <button 
                class="btn btn-disabled">
                Delete
            </button>
        </td>
    </tr>
    """

    return response


@app.route("/get-book-row/<int:id>", methods=["GET"])
def get_book_row(id):
    book = Book.query.get(id)
    author = Author.query.get(book.author_id)

    response = f"""
    <tr>
        <td>{book.title}</td>
        <td>{author.name}</td>
        <td>
            <button class="btn btn-primary"
                hx-get="/get-edit-form/{id}">
                Edit Title
            </button>
        </td>
        <td>
            <button hx-delete="/delete/{id}"
                class="btn btn-primary">
                Delete
            </button>
        </td>
    </tr>
    """

    return response


@app.route("/update/<int:id>", methods=["PUT"])
def update_book(id):
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
            <button class="btn btn-primary"
                hx-get="/get-edit-form/{id}">
                Edit Title
            </button>
        </td>
        <td>
            <button hx-delete="/delete/{id}"
                class="btn btn-primary">
                Delete
            </button>
        </td>
    </tr>
    """

    return response

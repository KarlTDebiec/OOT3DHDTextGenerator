#  Copyright 2020-2023 Karl T Debiec. All rights reserved. This software may be modified
#  and distributed under the terms of the BSD license. See the LICENSE file for details.


def get_model_book(db):
    class Book(db.Model):
        book_id = db.Column(db.Integer, primary_key=True)
        author_id = db.Column(db.Integer, db.ForeignKey("author.author_id"))
        title = db.Column(db.String)

    return Book

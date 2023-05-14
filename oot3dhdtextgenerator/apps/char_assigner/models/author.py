#  Copyright 2020-2023 Karl T Debiec. All rights reserved. This software may be modified
#  and distributed under the terms of the BSD license. See the LICENSE file for details.


def get_model_author(db):
    class Author(db.Model):
        author_id = db.Column(db.Integer, primary_key=True)
        name = db.Column(db.String)
        books = db.relationship("Book", backref="author")

        def __repr__(self):
            return f"<Author: {self.books}>"

    return Author

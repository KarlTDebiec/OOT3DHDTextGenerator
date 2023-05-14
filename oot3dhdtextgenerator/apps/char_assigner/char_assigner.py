#  Copyright 2020-2023 Karl T Debiec. All rights reserved. This software may be modified
#  and distributed under the terms of the BSD license. See the LICENSE file for details.

from os import getenv
from pathlib import Path
from typing import Any

from flask import Flask
from flask_sqlalchemy import SQLAlchemy

from oot3dhdtextgenerator.apps.char_assigner.models.author import get_model_author
from oot3dhdtextgenerator.apps.char_assigner.models.book import get_model_book
from oot3dhdtextgenerator.apps.char_assigner.routes.books import route_books
from oot3dhdtextgenerator.apps.char_assigner.routes.index import route_index


class CharAssigner:
    def __init__(self, **kwargs: Any) -> None:
        self.app = Flask(__name__)
        self.persistent_path = Path(
            getenv("PERSISTANT_STORATE_DIR", Path(__file__).parent)
        )
        self.db_path = self.persistent_path / "sqlite.db"

        self.app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{self.db_path}"
        self.app.config["SQLALCHEMY_ECHO"] = False
        self.app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

        self.db = SQLAlchemy(self.app)

        self.app.Author = get_model_author(self.db)
        self.app.Book = get_model_book(self.db)

        route_index(self.app, self.db)
        route_books(self.app, self.db)

        with self.app.app_context():
            self.db.create_all()

    def run(self, **kwargs: Any) -> None:
        self.app.run(**kwargs)

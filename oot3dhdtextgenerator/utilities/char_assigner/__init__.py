#  Copyright 2020-2023 Karl T Debiec. All rights reserved. This software may be modified
#  and distributed under the terms of the BSD license. See the LICENSE file for details.
from __future__ import annotations

from os import getenv
from pathlib import Path

from flask import Flask
from flask_sqlalchemy import SQLAlchemy

persistent_path = Path(getenv("PERSISTANT_STORATE_DIR", Path(__file__).parent))

app = Flask(__name__)

db_path = persistent_path / "sqlite.db"

app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{db_path}"
app.config["SQLALCHEMY_ECHO"] = False
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy()

from oot3dhdtextgenerator.utilities.char_assigner import models  # noqa: E402
from oot3dhdtextgenerator.utilities.char_assigner import views  # noqa: E402

db.init_app(app)

with app.app_context():
    db.create_all()

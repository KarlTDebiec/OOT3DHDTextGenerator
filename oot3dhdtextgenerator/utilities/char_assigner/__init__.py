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

db.init_app(app)

with app.app_context():
    db.create_all()

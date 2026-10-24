"""
app.py -- Flask entrypoint (Vercel looks for a top-level `app` here).
Static files live in public/ (served by Vercel's CDN; Flask serves them too when run locally).
"""
import os

from flask import Flask

app = Flask(__name__, static_folder="public", static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = int(float(os.environ.get("MAX_UPLOAD_MB", 4)) * 1024 * 1024) + 64 * 1024

import routes  # noqa: E402,F401  (registers the routes)

if __name__ == "__main__":
    # `python app.py` runs this file as __main__, so re-import it under its real name:
    # routes.py attaches to that copy of `app`, not to this one.
    from app import app as _app
    _app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 5000)),
             debug=os.environ.get("FLASK_DEBUG") == "1")

"""
app.py
------
The Flask application instance. routes.py does `from app import app`,
but this file didn't exist in the repo before -- that import would
have failed immediately on startup.
"""

from flask import Flask

app = Flask(__name__, static_folder="static", template_folder="templates")

import routes  # noqa: E402,F401  (import at the bottom registers the routes)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

"""
run.py
------
Entry point: run the server with `python3 run.py`, not `python3 app.py`.

Why a separate file: routes.py does `from app import app` to attach its
routes to the shared Flask instance. If app.py were executed directly
(`python3 app.py`), Python would import that file twice under two
different module names -- once as `__main__`, once as `app` (triggered
by routes.py's import) -- creating two SEPARATE Flask objects. Routes
would register on one; `.run()` would start the other; every request
would 404 even though the routes clearly exist.

Running this file instead means app.py is only ever imported the
normal way, so there's exactly one Flask instance everywhere.
"""

from app import app
import routes  # noqa: F401  (imported for its side effect: registering routes on `app`)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

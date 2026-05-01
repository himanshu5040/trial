import os
from datetime import timedelta
from time import perf_counter, time

from flask import Flask, redirect, render_template, request, session, url_for

from generator import DEFAULT_REASONING_PATHS, generate_reasoning, get_model_label
from selector import select_best

app = Flask(__name__)
# Fresh secret on each app start so old browser sessions stop working after restart.
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or os.urandom(32)
app.config.update(
    PERMANENT_SESSION_LIFETIME=timedelta(minutes=10),
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
)

DEMO_USERNAME = "admin"
DEMO_PASSWORD = "admin123"
USERS = {
    DEMO_USERNAME: DEMO_PASSWORD,
}


def _format_error(exc):
    message = str(exc)
    lowered = message.lower()

    if "failed to connect" in lowered or "connection refused" in lowered:
        return (
            "Could not reach Ollama from this deployed app. "
            "On Render, localhost points to the Render server, not your laptop. "
            "Set an OLLAMA_HOST environment variable to a reachable Ollama server, "
            "or deploy Ollama on the same server/VPS."
        )

    return f"Request failed: {exc}"


@app.before_request
def enforce_session_timeout():
    if request.endpoint in {"login", "register", "static"}:
        return None

    if not session.get("logged_in"):
        return redirect(url_for("login"))

    now = time()
    last_activity = session.get("last_activity")
    timeout_seconds = int(app.permanent_session_lifetime.total_seconds())

    if last_activity and now - last_activity > timeout_seconds:
        session.clear()
        return redirect(url_for("login"))

    session.permanent = True
    session["last_activity"] = now
    return None


@app.after_request
def disable_caching(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/login", methods=["GET", "POST"])
@app.route("/signin", methods=["GET", "POST"])
def login():
    if session.get("logged_in"):
        return redirect(url_for("index"))

    error = None

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if USERS.get(username) == password:
            session.clear()
            session.permanent = True
            session["logged_in"] = True
            session["username"] = username
            session["last_activity"] = time()
            return redirect(url_for("index"))

        error = "Invalid username or password."

    return render_template("login.html", error=error)


@app.route("/register", methods=["GET", "POST"])
@app.route("/signup", methods=["GET", "POST"])
def register():
    if session.get("logged_in"):
        return redirect(url_for("index"))

    error = None

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not username or not password or not confirm_password:
            error = "Please fill in all fields."
        elif username in USERS:
            error = "Username already exists."
        elif password != confirm_password:
            error = "Passwords do not match."
        else:
            USERS[username] = password
            return redirect(url_for("login"))

    return render_template("register.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    best = None
    question = ""
    answer_mode = "fast"
    error = None
    metrics = None

    if request.method == "POST":
        question = request.form["question"].strip()
        answer_mode = request.form.get("answer_mode", "fast")

        if question:
            started_at = perf_counter()

            try:
                paths = generate_reasoning(
                    question,
                    k=DEFAULT_REASONING_PATHS,
                    mode=answer_mode,
                )
                results, best = select_best(paths)
                metrics = {
                    "path_count": len(results),
                    "elapsed_seconds": round(perf_counter() - started_at, 2),
                    "answer_mode": answer_mode.title(),
                }
            except Exception as exc:
                error = _format_error(exc)
            finally:
                # One login is valid for only one model run.
                session.clear()
        else:
            error = "Please enter a question."

    return render_template(
        "index.html",
        results=results,
        best=best,
        question=question,
        answer_mode=answer_mode,
        error=error,
        metrics=metrics,
        model_label=get_model_label(),
    )


if __name__ == "__main__":
    app.run(debug=True)
    

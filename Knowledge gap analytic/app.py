import json
import time
from functools import wraps

from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash

from utils import db
from utils.question_bank import SUBJECTS, get_questions_by_subject, get_question
from utils.preprocess import score_attempt, DIFFICULTY_MAP
from utils.predict import predict_competency_and_score
from utils.recommendations import compute_weak_topics, build_recommendations

app = Flask(__name__)
app.secret_key = "dev-secret-key-change-this-in-production"  # move to env var in production

db.init_db()

QUESTIONS_PER_TEST = 5


# ---------------- Auth helpers ----------------

def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to continue.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapper


@app.context_processor
def inject_user():
    return {"current_user_name": session.get("user_name")}


# ---------------- Public routes ----------------

@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")
        security_answer = request.form.get("security_answer", "").strip().lower()

        if not name or not email or not password:
            flash("Please fill in all required fields.", "danger")
        elif password != confirm:
            flash("Passwords do not match.", "danger")
        elif len(password) < 6:
            flash("Password must be at least 6 characters.", "danger")
        else:
            ok = db.create_user(
                name, email,
                generate_password_hash(password),
                generate_password_hash(security_answer or "answer"),
            )
            if ok:
                flash("Account created! Please log in.", "success")
                return redirect(url_for("login"))
            else:
                flash("An account with that email already exists.", "danger")

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = db.get_user_by_email(email)
        if user and check_password_hash(user["password_hash"], password):
            session["user_id"] = user["id"]
            session["user_name"] = user["name"]
            return redirect(url_for("dashboard"))
        flash("Invalid email or password.", "danger")

    return render_template("login.html")


@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    step = request.form.get("step", "find")

    if request.method == "POST":
        if step == "find":
            email = request.form.get("email", "").strip().lower()
            user = db.get_user_by_email(email)
            if user:
                return render_template("forgot_password.html", step="reset", email=email)
            flash("No account found with that email.", "danger")

        elif step == "reset":
            email = request.form.get("email", "").strip().lower()
            answer = request.form.get("security_answer", "").strip().lower()
            new_password = request.form.get("new_password", "")
            user = db.get_user_by_email(email)
            if user and check_password_hash(user["security_answer_hash"], answer):
                if len(new_password) < 6:
                    flash("New password must be at least 6 characters.", "danger")
                    return render_template("forgot_password.html", step="reset", email=email)
                db.update_password(user["id"], generate_password_hash(new_password))
                flash("Password reset successfully. Please log in.", "success")
                return redirect(url_for("login"))
            flash("Security answer incorrect.", "danger")
            return render_template("forgot_password.html", step="reset", email=email)

    return render_template("forgot_password.html", step="find")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


# ---------------- Dashboard ----------------

@app.route("/dashboard")
@login_required
def dashboard():
    summary = db.get_dashboard_summary(session["user_id"])
    return render_template("dashboard.html", summary=summary, subjects=SUBJECTS)


# ---------------- Mock Test ----------------

@app.route("/test/select")
@login_required
def test_select():
    return render_template("select_subject.html", subjects=SUBJECTS)


@app.route("/test/<subject>")
@login_required
def test_take(subject):
    if subject not in SUBJECTS:
        flash("Unknown subject.", "danger")
        return redirect(url_for("test_select"))
    questions = get_questions_by_subject(subject, limit=QUESTIONS_PER_TEST)
    if not questions:
        flash("No questions available for this subject yet.", "warning")
        return redirect(url_for("test_select"))
    return render_template("test.html", subject=subject, questions=questions,
                            start_time=int(time.time()))


@app.route("/test/<subject>/submit", methods=["POST"])
@login_required
def test_submit(subject):
    questions = get_questions_by_subject(subject, limit=QUESTIONS_PER_TEST)
    answers = {}
    for q in questions:
        selected = request.form.get(f"q{q['id']}")
        if selected:
            answers[str(q["id"])] = selected

    start_time = int(request.form.get("start_time", int(time.time())))
    time_taken = max(1, int(time.time()) - start_time)

    result = score_attempt(questions, answers)

    # Previous score = accuracy of the user's last attempt (or current accuracy if first ever)
    prior_attempts = db.get_attempts_for_user(session["user_id"], limit=1)
    previous_score = prior_attempts[0]["accuracy"] if prior_attempts else result["accuracy"]

    avg_time_per_q = round(time_taken / result["total"], 1)
    topic_accuracy_overall = round(
        sum(result["topic_scores"].values()) / len(result["topic_scores"]), 1
    ) if result["topic_scores"] else result["accuracy"]

    competency, predicted_next_score, success_probability = predict_competency_and_score(
        accuracy=result["accuracy"],
        previous_score=previous_score,
        avg_time=avg_time_per_q,
        difficulty_level=result["avg_difficulty"],
        topic_accuracy=topic_accuracy_overall,
    )

    weak_topics = compute_weak_topics(result["topic_scores"])
    prior_count = len(db.get_attempts_for_user(session["user_id"]))
    recommendations = build_recommendations(weak_topics, next_mock_number=prior_count + 2)

    attempt_id = db.save_attempt(
        user_id=session["user_id"],
        subject=subject,
        total=result["total"],
        correct=result["correct"],
        accuracy=result["accuracy"],
        time_taken=time_taken,
        avg_time=avg_time_per_q,
        difficulty_level=result["avg_difficulty"],
        topic_scores=result["topic_scores"],
        competency=competency,
        predicted_next_score=predicted_next_score,
        success_probability=success_probability,
        weak_topics=weak_topics,
        recommendations=recommendations,
    )

    return redirect(url_for("result", attempt_id=attempt_id))


# ---------------- Performance Analysis / Result ----------------

@app.route("/result/<int:attempt_id>")
@login_required
def result(attempt_id):
    attempt = db.get_attempt(attempt_id)
    if not attempt or attempt["user_id"] != session["user_id"]:
        flash("Result not found.", "danger")
        return redirect(url_for("dashboard"))

    topic_scores = json.loads(attempt["topic_scores"])
    weak_topics = json.loads(attempt["weak_topics"])
    recommendations = json.loads(attempt["recommendations"])

    return render_template(
        "result.html",
        attempt=attempt,
        topic_scores=topic_scores,
        weak_topics=weak_topics,
        recommendations=recommendations,
    )


# ---------------- Personalized Recommendation (standalone view) ----------------

@app.route("/recommendation/<int:attempt_id>")
@login_required
def recommendation(attempt_id):
    attempt = db.get_attempt(attempt_id)
    if not attempt or attempt["user_id"] != session["user_id"]:
        flash("Result not found.", "danger")
        return redirect(url_for("dashboard"))

    weak_topics = json.loads(attempt["weak_topics"])
    recommendations = json.loads(attempt["recommendations"])

    return render_template(
        "recommendation.html",
        attempt=attempt,
        weak_topics=weak_topics,
        recommendations=recommendations,
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

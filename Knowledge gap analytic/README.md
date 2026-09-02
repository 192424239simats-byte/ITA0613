# ExamGap Analytics
Machine Learning-Driven Knowledge Gap Analytics and Competency Assessment for Competitive Examination Aspirants

A working Flask prototype implementing all 8 modules from the spec: auth, dashboard,
mock test, performance analysis, Random Forest competency assessment, XGBoost
next-score prediction, knowledge gap analytics, and personalized recommendations.

## Setup

```bash
cd KnowledgeGapAnalytics
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Generate the training dataset and train both models (only needed once,
# or whenever you want to regenerate them)
python dataset/generate_data.py
python models/train_model.py

# Run the app
python app.py
```

Then open **http://localhost:5000**.

## How it works

1. **Register / Log in** (Module 1) — accounts stored in `database/students.db`
   (SQLite). Passwords are hashed with Werkzeug; "Forgot Password" uses a
   security-question flow.
2. **Dashboard** (Module 2) — shows tests taken, overall accuracy, current
   competency (a "stamped" badge), predicted next score, topic-accuracy bars,
   recent attempts, and the latest recommendation.
3. **Mock Test** (Module 3) — pick a subject, answer 5 MCQs styled as OMR
   bubbles, submit. Question bank lives in `utils/question_bank.py` — swap this
   for a `questions` DB table if you want an admin-editable bank later.
4. **Performance Analysis** (Module 4) — total score, accuracy, time taken
   (measured server-side from a hidden `start_time` field), subject/topic-wise
   breakdown.
5. **Random Forest** (Module 5) — `models/train_model.py` trains a
   `RandomForestClassifier` on `[accuracy, previous_score, avg_time,
   difficulty_level, topic_accuracy]` to predict competency
   (Beginner/Intermediate/Advanced). Saved to `models/random_forest.pkl`.
6. **XGBoost** (Module 6) — an `XGBRegressor` on the same features predicts the
   next score (0–100). "Probability of success" is derived from a blend of the
   predicted score and the Random Forest's class confidence (see
   `utils/predict.py` — this is a reasonable, explainable composite rather than
   a separately-trained probability model; swap in a dedicated classifier if
   you want a purer estimate). Saved to `models/xgboost.pkl`.
7. **Knowledge Gap Analytics** (Module 7) — any topic scoring below 60% on an
   attempt is flagged as a gap (`utils/recommendations.py:WEAK_THRESHOLD`).
8. **Personalized Recommendation** (Module 8) — rule-based mapping from weak
   topics to concrete actions (`utils/recommendations.py:TOPIC_ACTIONS`), plus
   a nudge to attempt the next mock test.

## Notes on this environment vs. your machine

This prototype was built in a sandbox without internet access, so the
`xgboost` package couldn't be installed here — `models/train_model.py`
automatically falls back to `sklearn.ensemble.GradientBoostingRegressor`
(same role: gradient-boosted regression) when `xgboost` isn't importable, so
the app still runs end-to-end. On your machine, `pip install -r
requirements.txt` installs real `xgboost`, and the script will use it
automatically — no code changes needed.

## Extending this prototype

- **Database**: swap `sqlite3` calls in `utils/db.py` for a MySQL connector
  (e.g. `mysql-connector-python` or `SQLAlchemy`) — it's the only file that
  touches the DB.
- **Question bank**: move `utils/question_bank.py` into a `questions` table
  and add an admin UI to manage it.
- **Auth**: swap the hand-rolled session auth for `Flask-Login` if you want
  "remember me", role-based access, etc.
- **Retraining**: `models/train_model.py` is decoupled from the app — retrain
  on real attempt data exported from `database/students.db` as it accumulates.

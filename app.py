from flask import Flask, render_template, request, redirect, url_for, session
import sqlite3
import json
from flask_bcrypt import Bcrypt
import requests
import os
from dotenv import load_dotenv
import numpy as np

load_dotenv()

############################################
# NBA Prediction Model Code
############################################

def streak_effect(streak_length, decay_rate=0.1):
    """Calculate the streak effect using exponential decay."""
    return 1 - np.exp(-decay_rate * streak_length)

def convert_defensive_rank_to_rating(rank):
    """Convert defense rank (1-30) to defensive rating (0-1)."""
    return 1 - (rank - 1) / 29

def normalize_weights(weights):
    """Normalize weights so they sum to 1."""
    total_weight = sum(weights.values())
    if total_weight == 0:
        return weights
    return {key: value / total_weight for key, value in weights.items()}

def calculate_likelihood(D_vsPos, P_24_25, P_H2H, P_L5, P_L10, P_L20, P_23_24, 
                         Streak_Pos, Streak_Neg, weights):
    """Combine all factors into a likelihood score."""
    return (D_vsPos * weights['D_vsPos'] +
            P_24_25 * weights['P_24_25'] +
            P_H2H * weights['P_H2H'] +
            P_L5 * weights['P_L5'] +
            P_L10 * weights['P_L10'] +
            P_L20 * weights['P_L20'] +
            P_23_24 * weights['P_23_24'] +
            Streak_Pos * weights['Streak_Pos'] -
            Streak_Neg * weights['Streak_Neg'])

def convert_to_probability(likelihood, scale_factor=1.0):
    """Convert likelihood to probability via a scaled sigmoid."""
    # Cap the likelihood to avoid extreme exponent
    likelihood = min(max(likelihood, -3), 3)
    return 1 / (1 + np.exp(-likelihood / scale_factor))

def get_player_prediction(team_defensive_rank, player_position, player_stats, streaks, weights, line, player_average):
    """Main function to get player prediction & projection."""
    D_vsPos = convert_defensive_rank_to_rating(team_defensive_rank)

    # Adjust D_vsPos based on position
    position_map = {'PG': 11, 'SG': 10, 'SF': 4, 'PF': 6, 'C': 10}
    rank_vs_pos = position_map.get(player_position, team_defensive_rank)
    D_vsPos = convert_defensive_rank_to_rating(rank_vs_pos)

    # Extract stats
    P_24_25 = player_stats['24/25']
    P_H2H = player_stats['H2H']
    P_L5 = player_stats['L5']
    P_L10 = player_stats['L10']
    P_L20 = player_stats['L20']
    P_23_24 = player_stats['23/24']

    # Calculate streak effects
    Streak_Pos = streak_effect(streaks['Positive'])
    Streak_Neg = streak_effect(streaks['Negative'])

    # Normalize weights & compute likelihood
    normalized_weights = normalize_weights(weights)
    likelihood = calculate_likelihood(D_vsPos, P_24_25, P_H2H, P_L5, P_L10, P_L20,
                                      P_23_24, Streak_Pos, Streak_Neg, normalized_weights)

    # Probability
    probability = convert_to_probability(likelihood)
    probability_to_hit_line = probability * 100
    is_over_line = "Yes" if probability_to_hit_line > 50 else "No"

    # Simple projected stat
    projected_stat = player_average * (1 + min(max(likelihood * 0.05, -0.1), 0.1))

    return {
        "Likelihood Score": round(likelihood, 2),
        "Probability to Exceed Line (%)": round(probability_to_hit_line, 2),
        "Will Exceed Line": is_over_line,
        "Projected Stat (e.g., points)": round(projected_stat, 1)
    }

############################################
# Win/Loss Determination
############################################

def determine_win_loss(predicted_yes_no, user_line, actual_outcome):
    """
    Return "Win" if the user guessed correctly, "Loss" otherwise.
    predicted_yes_no: "Yes" or "No"
    user_line: e.g., 7.5
    actual_outcome: e.g., 8.0 means they exceeded the line
    """
    actually_exceeded = (actual_outcome > user_line)
    predicted_yes = (predicted_yes_no == "Yes")
    if actually_exceeded == predicted_yes:
        return "Win"
    else:
        return "Loss"


############################################
# Flask App & User Auth
############################################

app = Flask(__name__)
app.secret_key = "SUPER_SECRET_KEY"  # Replace with secure key in production
bcrypt = Bcrypt(app)
DATABASE = "users.db"

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def get_user_by_username(username):
    db = get_db()
    user = db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    db.close()
    return user

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        existing_user = get_user_by_username(username)
        if existing_user:
            return "Username already taken. <a href='/register'>Try another</a>"

        hashed_pw = bcrypt.generate_password_hash(password).decode("utf-8")

        # Optional user defaults
        default_weights = {
            "D_vsPos": 0.25,
            "P_24_25": 0.20,
            "P_H2H": 0.15,
            "P_L5": 0.10,
            "P_L10": 0.10,
            "P_L20": 0.05,
            "P_23_24": 0.05,
            "Streak_Pos": 0.10,
            "Streak_Neg": 0.10
        }
        default_weights_str = json.dumps(default_weights)

        db = get_db()
        db.execute("""
            INSERT INTO users (username, password, default_weights)
            VALUES (?, ?, ?)
        """, (username, hashed_pw, default_weights_str))
        db.commit()
        db.close()

        return redirect(url_for("login"))

    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        user = get_user_by_username(username)

        if user:
            if bcrypt.check_password_hash(user["password"], password):
                session["user_id"] = user["id"]
                session["username"] = user["username"]
                return redirect(url_for("index"))
            else:
                return "Incorrect password. <a href='/login'>Try again</a>."
        else:
            return "Username not found. <a href='/register'>Register here</a>."

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))

@app.route("/")
def index():
    user_id = session.get("user_id")
    if user_id:
        db = get_db()
        user = db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        db.close()
        user_defaults = json.loads(user["default_weights"])
        return render_template("index.html", username=user["username"], defaults=user_defaults)
    else:
        return render_template("index.html", username=None, defaults=None)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    user_id = session.get("user_id")  # If you require a user to be logged in
    if not user_id:
        return redirect(url_for("login"))

    if request.method == "POST":
        # 1) Parse all form field
        player_name = request.form["player_name"]
        team_defensive_rank = int(request.form["team_defensive_rank"])
        player_position = request.form["player_position"]
        player_average = float(request.form["player_average"])
        
        # Example: Player stats
        player_stats = {
            "24/25": float(request.form["P_24_25"]),
            "H2H": float(request.form["P_H2H"]),
            "L5": float(request.form["P_L5"]),
            "L10": float(request.form["P_L10"]),
            "L20": float(request.form["P_L20"]),
            "23/24": float(request.form["P_23_24"])
        }

        # Streaks
        streaks = {
            "Positive": float(request.form["streak_Pos"]),
            "Negative": float(request.form["streak_Neg"])
        }

        # Weights
        weights = {
        'D_vsPos': float(request.form['D_vsPos']),
        'P_24_25': float(request.form['P_24_25']),
        'P_H2H': float(request.form['P_H2H']),
        'P_L5': float(request.form['P_L5']),
        'P_L10': float(request.form['P_L10']),
        'P_L20': float(request.form['P_L20']),
        'P_23_24': float(request.form['P_23_24']),
        'Streak_Pos': float(request.form['Streak_Pos']),
        'Streak_Neg': float(request.form['Streak_Neg'])
        }

        # Betting line
        user_line = float(request.form["user_line"])

        # 2) Run your prediction logic (Example)
        # Suppose you have a function: get_player_prediction(...)
        # that returns a dict of { "Likelihood Score", "Probability to Exceed Line (%)", "Projected Stat", etc. }
        prediction = get_player_prediction(
            team_defensive_rank,
            player_position,
            player_stats,
            streaks,
            weights,
            line=user_line,
            player_average=player_average
        )

        # 3) (Optional) Store the predictions in your DB if desired
        db = get_db()
        db.execute("""
            INSERT INTO predictions (
                user_id,
                player_name,
                team_defensive_rank,
                player_position,
                player_stats,
                streaks,
                weights,
                probability_exceed_line,
                will_exceed_line,
                projected_stat,
                likelihood_score,
                user_line
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            player_name,
            team_defensive_rank,
            player_position,
            json.dumps(player_stats),
            json.dumps(streaks),
            json.dumps(weights),
            prediction["Probability to Exceed Line (%)"],
            prediction["Will Exceed Line"],
            prediction["Projected Stat (e.g., points)"],
            prediction["Likelihood Score"],
            user_line
        ))
        db.commit()
        db.close()

        # 4) Return a results page (render a template or redirect)
        return render_template("results.html", prediction=prediction, player_name=player_name)

    else:
        # If GET: Just show the predict form
        return render_template("predict.html")

@app.route("/history")
def history():
    user_id = session.get("user_id")
    if not user_id:
        return redirect(url_for("login"))

    db = get_db()
    rows = db.execute(
        """
        SELECT
            id,
            player_name,
            timestamp,
            team_defensive_rank,
            player_position,
            probability_exceed_line,
            will_exceed_line,
            projected_stat,
            likelihood_score,
            actual_outcome,
            user_line,
            win_loss
        FROM predictions
        WHERE user_id = ?
        ORDER BY timestamp DESC
        """,
        (user_id,)
    ).fetchall()
    db.close()

    return render_template("history.html", predictions=rows)

@app.route("/update_outcome/<int:prediction_id>", methods=["GET", "POST"])
def update_outcome(prediction_id):
    """Allows user to set or update the 'actual_outcome' + compute Win/Loss."""
    user_id = session.get("user_id")
    if not user_id:
        return redirect(url_for("login"))

    db = get_db()
    # 1. Check ownership
    pred_row = db.execute(
        "SELECT * FROM predictions WHERE id = ? AND user_id = ?",
        (prediction_id, user_id)
    ).fetchone()

    if not pred_row:
        db.close()
        return "No matching prediction found or not authorized."

    if request.method == "POST":
        actual_outcome = float(request.form["actual_outcome"])

        # Retrieve line and predicted yes/no from the record
        user_line = pred_row["user_line"]  # float
        predicted_yes_no = pred_row["will_exceed_line"]  # "Yes" or "No"

        # Determine win/loss
        result = determine_win_loss(predicted_yes_no, user_line, actual_outcome)

        db.execute(
            "UPDATE predictions SET actual_outcome = ?, win_loss = ? WHERE id = ? AND user_id = ?",
            (actual_outcome, result, prediction_id, user_id)
        )
        db.commit()
        db.close()
        return redirect(url_for("history"))
    else:
        db.close()
        return render_template("update_outcome.html", prediction_id=prediction_id)
    
@app.route("/delete_prediction/<int:prediction_id>", methods=["GET", "POST"])
def delete_prediction(prediction_id):
    user_id = session.get("user_id")
    if not user_id:
        return redirect(url_for("login"))

    db = get_db()
    # 1. Verify the prediction belongs to this user
    pred_row = db.execute(
        "SELECT * FROM predictions WHERE id = ? AND user_id = ?",
        (prediction_id, user_id)
    ).fetchone()
    if not pred_row:
        db.close()
        return "No matching prediction found or not authorized."

    if request.method == "POST":
        # 2. Perform the deletion
        db.execute(
            "DELETE FROM predictions WHERE id = ? AND user_id = ?",
            (prediction_id, user_id)
        )
        db.commit()
        db.close()

        return redirect(url_for("history"))
    else:
        # Render a confirmation page
        db.close()
        return render_template("delete_confirm.html", prediction=pred_row)
    
@app.route("/trends")
def trends():
    user_id = session.get("user_id")
    if not user_id:
        return redirect(url_for("login"))

    db = get_db()
    rows = db.execute("""
        SELECT
            timestamp,
            projected_stat,
            actual_outcome
        FROM predictions
        WHERE user_id = ?
        ORDER BY timestamp ASC
    """, (user_id,)).fetchall()
    db.close()

    # Convert rows to a list of dict for easy JSON usage
    data_points = []
    for row in rows:
        # row['timestamp'] might be a string or datetime; we can store as string
        data_points.append({
            "timestamp": str(row["timestamp"]),  # e.g. "2024-01-01 20:00:00"
            "predicted": float(row["projected_stat"]) if row["projected_stat"] else None,
            "actual": float(row["actual_outcome"]) if row["actual_outcome"] else None
        })

    return render_template("trends.html", data_points=data_points)

########################################
# Metrics
########################################
@app.route("/metrics")
def metrics():
    user_id = session.get("user_id")
    if not user_id:
        return redirect(url_for("login"))

    db = get_db()
    rows = db.execute(
        """
        SELECT
            win_loss,
            actual_outcome,
            projected_stat,
            probability_exceed_line
        FROM predictions
        WHERE user_id = ?
        """,
        (user_id,)
    ).fetchall()
    db.close()

    total_predictions = 0
    total_wins = 0
    total_losses = 0
    # (Optional) Sums for advanced metrics
    total_actual = 0.0
    total_projected = 0.0

    for row in rows:
        total_predictions += 1
        if row["win_loss"] == "Win":
            total_wins += 1
        elif row["win_loss"] == "Loss":
            total_losses += 1
        
        # Accumulate sums if you want averages
        if row["actual_outcome"] is not None:
            total_actual += float(row["actual_outcome"])
        if row["projected_stat"] is not None:
            total_projected += float(row["projected_stat"])

    if total_predictions > 0:
        win_rate = (total_wins / total_predictions) * 100
        avg_actual = total_actual / total_predictions
        avg_projected = total_projected / total_predictions
    else:
        win_rate = 0
        avg_actual = 0
        avg_projected = 0

    # Prepare any additional metrics you want:
    metrics_data = {
        "total_predictions": total_predictions,
        "total_wins": total_wins,
        "total_losses": total_losses,
        "win_rate": round(win_rate, 2),
    }

    return render_template("metrics.html", metrics=metrics_data)


########################################
# 4) Integrating SportsDataIO or Another API
########################################

def fetch_latest_player_stats():
    """
    Example: Fetch daily player stats from SportsDataIO.
    You can adapt the endpoint & parameters to your plan.
    """
    api_key = os.getenv("SPORTSDATAIO_KEY")
    if not api_key:
        raise ValueError("No SPORTSDATAIO_KEY found in environment (.env).")

    # Example endpoint: daily player stats for Jan 01, 2024
    endpoint = "https://api.sportsdata.io/v3/nba/stats/json/PlayerGameStatsByDate/2024-JAN-01"

    headers = {
        "Ocp-Apim-Subscription-Key": api_key
    }

    response = requests.get(endpoint, headers=headers)
    response.raise_for_status()
    data = response.json()  # List of dicts with stats
    return data

def store_player_stats_in_db(data):
    """
    Insert fetched stats into player_stats table.
    Adjust field names to match the actual API response.
    """
    db = get_db()
    for stat in data:
        player_id = stat.get("PlayerID", 0)
        player_name = stat.get("Name", "Unknown")
        points = stat.get("Points", 0.0)
        rebounds = stat.get("Rebounds", 0.0)
        assists = stat.get("Assists", 0.0)
        team = stat.get("Team", "UNK")
        date = stat.get("Date", "2024-01-01")  # or stat.get("Day") if provided

        db.execute("""
            INSERT INTO player_stats (player_id, player_name, points, rebounds, assists, team, date)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (player_id, player_name, points, rebounds, assists, team, date))
    db.commit()
    db.close()

@app.route("/fetch_data")
def fetch_data():
    """
    Manually fetch data from the external API and store in DB.
    Then link user to /players to see results.
    """
    user_id = session.get("user_id")
    if not user_id:
        return redirect(url_for("login"))

    # Optional: if you only want an admin user to do this,
    # check if user is admin here.

    data = fetch_latest_player_stats()
    store_player_stats_in_db(data)
    return "Data fetched! <a href='/players'>View Players</a>"

############################################
# NEW ROUTE /players
############################################

@app.route("/players")
def show_players():
    """
    Show the stats we fetched from the external API in a table.
    """
    user_id = session.get("user_id")
    if not user_id:
        return redirect(url_for("login"))

    db = get_db()
    rows = db.execute("""
        SELECT player_id, player_name, points, rebounds, assists, team, date
        FROM player_stats
        ORDER BY date DESC, points DESC
    """).fetchall()
    db.close()

    return render_template("players.html", players=rows)

if __name__ == "__main__":
    app.run(debug=True)

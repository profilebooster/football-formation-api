from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

LIVE_STORE = {}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

\\ DB_PATH = Path("Bundesliga_2023_2024.duckdb")
DB_PATH = Path("demo_bundesliga.duckdb")

MODEL_BUNDLE = joblib.load("rf_bundle.joblib")
pipe = MODEL_BUNDLE["pipe"]
num_cols = MODEL_BUNDLE["num_cols"]
cat_cols = MODEL_BUNDLE["cat_cols"]


def get_connection():
    return duckdb.connect(str(DB_PATH), read_only=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df["x_position"].values
    y = df["y_position"].values

    centroid_x = np.mean(x)
    centroid_y = np.mean(y)

    width = x.max() - x.min()
    length = y.max() - y.min()

    distances = np.sqrt((x - centroid_x) ** 2 + (y - centroid_y) ** 2)

    return pd.DataFrame([{
        "x_norm_mean": np.mean(x),
        "y_norm_mean": np.mean(y),
        "x_norm_std": np.std(x),
        "y_norm_std": np.std(y),
        "speed_mean": df["speed"].mean(),
        "dominant_possession": 1,
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "width": width,
        "length": length,
        "dist_mean": np.mean(distances),
        "dist_std": np.std(distances),
        "game_section": df["game_section"].mode().iloc[0]
        if "game_section" in df.columns and not df["game_section"].mode().empty
        else "firstHalf",
        "dominant_possession_team": "In_possession",
    }])

def flip_selected_team_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Für ein einzelnes Team:
    Wenn das Team in den ersten Frames im Durchschnitt rechts steht (x > 0),
    wird x gespiegelt, damit das Team von links nach rechts spielt.
    """
    if df.empty:
        return df.copy()

    flipped = df.copy()

    first_frames = flipped.nsmallest(1000, "frame")
    mean_x = first_frames["x_position"].mean()

    if mean_x > 0:
        flipped["x_position"] = flipped["x_position"] * -1

    return flipped

def build_players_for_plot(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []

    d = df.copy()
    d = d[d["player_name"].notna()]
    d = d[d["player_name"] != "BALL"]

    grouped = (
        d.groupby("player_name")
        .agg(
            x_norm_mean=("x_position", "mean"),
            y_norm_mean=("y_position", "mean"),
            x_norm_std=("x_position", "std"),
            y_norm_std=("y_position", "std"),
        )
        .reset_index()
    )

    grouped["x_norm_std"] = grouped["x_norm_std"].fillna(0.1)
    grouped["y_norm_std"] = grouped["y_norm_std"].fillna(0.1)

    # goalkeeper entfernen: linksester Spieler
    if len(grouped) > 10:
        gk_idx = grouped["x_norm_mean"].idxmin()
        grouped = grouped.drop(index=gk_idx)

    # maximal 10 Feldspieler
    grouped = grouped.head(10)

    return grouped.to_dict(orient="records")


@app.get("/")
def root():
    return {
        "status": "API läuft",
        "db_exists": DB_PATH.exists(),
        "db_path": str(DB_PATH),
    }


@app.get("/matches")
def get_matches():
    conn = get_connection()

    df = conn.execute("""
        SELECT DISTINCT match_id
        FROM tracking_raw_observed
        ORDER BY match_id
    """).fetchdf()

    conn.close()
    return df.to_dict(orient="records")


@app.get("/teams")
def get_teams(match_id: str | None = None):
    conn = get_connection()

    if match_id:
        query = """
            SELECT DISTINCT
                t.three_letter_code AS team_code
            FROM tracking_raw_observed tr
            LEFT JOIN teams t
                ON tr.team_id = t.id
            WHERE tr.match_id = ?
              AND t.three_letter_code IS NOT NULL
            ORDER BY team_code
        """
        df = conn.execute(query, [match_id]).fetchdf()
    else:
        query = """
            SELECT
                id,
                name,
                three_letter_code AS team_code
            FROM teams
            WHERE three_letter_code IS NOT NULL
            ORDER BY three_letter_code
        """
        df = conn.execute(query).fetchdf()

    conn.close()
    return df.to_dict(orient="records")


@app.get("/tracking/team-window")
def get_team_tracking_window(
    match_id: str = Query(...),
    team_code: str = Query(...),
    start_minute: int = Query(...),
    end_minute: int = Query(...),
    limit: int = Query(5000),
):
    conn = get_connection()

    query = """
        SELECT 
            tr.match_id,
            t.three_letter_code AS team_code,
            p.short_name AS player_name,
            tr.player_id,
            tr.team_id,
            tr.frame,
            tr.minute,
            tr.game_section,
            tr.x_position,
            tr.y_position,
            tr.speed,
            tr.ball_status,
            tr.ball_possession
        FROM tracking_raw_observed tr
        LEFT JOIN teams t 
            ON tr.team_id = t.id
        LEFT JOIN players p 
            ON tr.player_id = p.id
        WHERE tr.match_id = ?
          AND UPPER(t.three_letter_code) = UPPER(?)
          AND tr.minute >= ?
          AND tr.minute < ?
        ORDER BY tr.frame, tr.player_id
        LIMIT ?
    """

    df = conn.execute(
        query,
        [match_id, team_code, start_minute, end_minute, limit],
    ).fetchdf()

    conn.close()
    df = df.where(pd.notnull(df), None)
    return df.to_dict(orient="records")


@app.get("/formation/live")
def get_live_formation(match_id: str, team_code: str):
    key = f"{match_id}_{team_code}"

    if key not in LIVE_STORE:
        LIVE_STORE[key] = {
            "current_minute": 0,
            "all_data": pd.DataFrame(),
        }

    current_minute = LIVE_STORE[key]["current_minute"]
    start_minute = current_minute
    end_minute = current_minute + 2

    conn = get_connection()

    query = """
        SELECT
            tr.match_id,
            t.three_letter_code AS team_code,
            p.short_name AS player_name,
            tr.player_id,
            tr.team_id,
            tr.frame,
            tr.minute,
            tr.game_section,
            tr.x_position,
            tr.y_position,
            tr.speed,
            tr.ball_status,
            tr.ball_possession
        FROM tracking_raw_observed tr
        LEFT JOIN teams t
            ON tr.team_id = t.id
        LEFT JOIN players p
            ON tr.player_id = p.id
        WHERE tr.match_id = ?
          AND UPPER(t.three_letter_code) = UPPER(?)
          AND tr.minute >= ?
          AND tr.minute < ?
        ORDER BY tr.frame, tr.player_id
        LIMIT 5000
    """

    new_df = conn.execute(
        query,
        [match_id, team_code, start_minute, end_minute],
    ).fetchdf()

    conn.close()

    if new_df.empty:
        return {
            "message": "No more live data",
            "current_minute": current_minute,
        }

    new_df["x_position"] = new_df["x_position"].fillna(0)
    new_df["y_position"] = new_df["y_position"].fillna(0)
    new_df["speed"] = new_df["speed"].fillna(0)
    
    new_df["player_name"] = new_df["player_name"].fillna("UNKNOWN")
    new_df["team_code"] = new_df["team_code"].fillna(team_code)
    new_df["game_section"] = new_df["game_section"].fillna("UNKNOWN")

    LIVE_STORE[key]["all_data"] = pd.concat(
        [LIVE_STORE[key]["all_data"], new_df],
        ignore_index=True,
    )

    all_data = LIVE_STORE[key]["all_data"]

    # Spiegeln berücksichtigen
    processed_data = flip_selected_team_if_needed(all_data)
    
    features = build_features(processed_data)

    for c in num_cols:
        if c not in features.columns:
            features[c] = 0.0

    for c in cat_cols:
        if c not in features.columns:
            features[c] = "Unknown"

    X = features[num_cols + cat_cols]
    prediction = pipe.predict(X)[0]

    players_for_plot = build_players_for_plot(processed_data)

    LIVE_STORE[key]["current_minute"] += 2

    return {
        "match_id": match_id,
        "team_code": team_code,
        "loaded_minutes": f"{start_minute}-{end_minute}",
        "total_rows": len(all_data),
        "predicted_formation": str(prediction),
        "players": players_for_plot,
    }


@app.get("/formation/reset")
def reset_live_formation(match_id: str, team_code: str):
    key = f"{match_id}_{team_code}"

    if key in LIVE_STORE:
        del LIVE_STORE[key]

    return {
        "message": "Live store reset",
        "match_id": match_id,
        "team_code": team_code,
    }


@app.get("/debug/tables")
def debug_tables():
    conn = get_connection()
    df = conn.execute("SHOW TABLES").fetchdf()
    conn.close()
    return df.to_dict(orient="records")


@app.get("/debug/schema/{table_name}")
def debug_schema(table_name: str):
    conn = get_connection()
    df = conn.execute(f"DESCRIBE {table_name}").fetchdf()
    conn.close()
    return df.to_dict(orient="records")
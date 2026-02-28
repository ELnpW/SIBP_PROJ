import json
import math
import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# =========================
# CONFIG
# =========================
DATA_DIR = Path("data")
SQLITE_PATH = DATA_DIR / "database.sqlite"

OUT_DIR = Path("data_out")
OUT_DIR.mkdir(exist_ok=True)

LAST_N = 5
SEED = 561

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10  # test is remainder

# ELO params
ELO_BASE = 1500.0
ELO_K = 20.0
ELO_HOME_ADV = 60.0  # home advantage in Elo points (tunable)


# =========================
# Helpers
# =========================
def safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0


def outcome_label(home_goals: int, away_goals: int) -> int:
    # 0 home win, 1 draw, 2 away win
    if home_goals > away_goals:
        return 0
    if home_goals == away_goals:
        return 1
    return 2


def expected_score(rating_a: float, rating_b: float) -> float:
    # standard Elo expectation
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def update_elo(r_home: float, r_away: float, home_goals: int, away_goals: int) -> Tuple[float, float]:
    # actual score
    if home_goals > away_goals:
        s_home, s_away = 1.0, 0.0
    elif home_goals == away_goals:
        s_home, s_away = 0.5, 0.5
    else:
        s_home, s_away = 0.0, 1.0

    # apply home advantage only for expectation
    e_home = expected_score(r_home + ELO_HOME_ADV, r_away)
    e_away = 1.0 - e_home

    new_home = r_home + ELO_K * (s_home - e_home)
    new_away = r_away + ELO_K * (s_away - e_away)
    return new_home, new_away


@dataclass
class TeamForm:
    W: int = 0
    D: int = 0
    L: int = 0
    GF: int = 0
    GA: int = 0
    seq: str = ""  # e.g. "LDWLL"

    def add(self, result_char: str, gf: int, ga: int):
        if result_char == "W":
            self.W += 1
        elif result_char == "D":
            self.D += 1
        else:
            self.L += 1
        self.GF += gf
        self.GA += ga
        self.seq += result_char

    def finalize(self) -> "TeamForm":
        # keep seq length LAST_N
        self.seq = self.seq[-LAST_N:]
        return self

    def extra_stats(self) -> Tuple[int, int, float, float]:
        pts = self.W * 3 + self.D
        gd = self.GF - self.GA
        avg_gf = round(safe_div(self.GF, LAST_N), 2)
        avg_ga = round(safe_div(self.GA, LAST_N), 2)
        return pts, gd, avg_gf, avg_ga


def form_to_text(prefix: str, f: TeamForm) -> str:
    pts, gd, avg_gf, avg_ga = f.extra_stats()
    return (
        f"{prefix} last{LAST_N}: W{f.W} D{f.D} L{f.L}, "
        f"GF {f.GF} GA {f.GA}, GD {gd}, PTS {pts}, AvgGF {avg_gf} AvgGA {avg_ga}, seq {f.seq}."
    )


def build_example_text(
    league_name: str,
    season: str,
    home_team: str,
    away_team: str,
    home_last: TeamForm,
    away_last: TeamForm,
    home_home_last: TeamForm,
    away_away_last: TeamForm,
    home_elo: float,
    away_elo: float,
) -> str:
    # Same “style” as before, only richer
    # Keep it readable and consistent.
    return (
        f"League: {league_name}. Season: {season}. "
        f"Home team: {home_team}. Away team: {away_team}. "
        f"Home ELO: {round(home_elo, 1)}. Away ELO: {round(away_elo, 1)}. "
        f"{form_to_text('Home', home_last)} "
        f"{form_to_text('Away', away_last)} "
        f"{form_to_text('Home home', home_home_last)} "
        f"{form_to_text('Away away', away_away_last)}"
    )


# =========================
# Data extraction
# =========================
def fetch_matches(conn: sqlite3.Connection) -> List[dict]:
    """
    Reads from Kaggle European Soccer Database (database.sqlite).
    Pulls match rows with team/league names.

    Uses Match table columns:
      date, season, league_id, home_team_api_id, away_team_api_id,
      home_team_goal, away_team_goal
    plus Team/League names.
    """
    q = """
    SELECT
        m.date as date,
        m.season as season,
        l.name as league_name,
        ht.team_long_name as home_team,
        at.team_long_name as away_team,
        m.home_team_api_id as home_id,
        m.away_team_api_id as away_id,
        m.home_team_goal as home_goals,
        m.away_team_goal as away_goals
    FROM Match m
    JOIN League l ON l.id = m.league_id
    JOIN Team ht ON ht.team_api_id = m.home_team_api_id
    JOIN Team at ON at.team_api_id = m.away_team_api_id
    WHERE m.home_team_goal IS NOT NULL AND m.away_team_goal IS NOT NULL
      AND m.date IS NOT NULL
    ORDER BY m.date ASC
    """
    cur = conn.cursor()
    rows = cur.execute(q).fetchall()
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in rows]


def last_n_form(history: List[Tuple[str, int, int]]) -> TeamForm:
    """
    history: list of tuples (result_char, gf, ga) for that team in chronological order.
    We take last N.
    """
    f = TeamForm()
    for res, gf, ga in history[-LAST_N:]:
        f.add(res, gf, ga)
    return f.finalize()


def main():
    random.seed(SEED)

    if not SQLITE_PATH.exists():
        raise FileNotFoundError(f"Missing {SQLITE_PATH}. Put database.sqlite in data/ folder.")

    conn = sqlite3.connect(str(SQLITE_PATH))
    matches = fetch_matches(conn)
    print(f"Loaded matches: {len(matches)}")

    # Histories
    # overall_history[team_id] -> list of (W/D/L, gf, ga)
    overall_history: Dict[int, List[Tuple[str, int, int]]] = {}
    home_history: Dict[int, List[Tuple[str, int, int]]] = {}   # only matches when team played at home
    away_history: Dict[int, List[Tuple[str, int, int]]] = {}   # only matches when team played away

    # ELO per team (global). You can also reset per season/league, but global works fine & simple.
    elo: Dict[int, float] = {}

    examples = []

    for m in matches:
        date = (m["date"] or "")[:10]
        season = m["season"] or "unknown"
        league_name = m["league_name"] or "unknown"

        home_id = int(m["home_id"])
        away_id = int(m["away_id"])
        home_team = m["home_team"]
        away_team = m["away_team"]
        hg = int(m["home_goals"])
        ag = int(m["away_goals"])

        # Need at least LAST_N previous matches in overall history for both teams
        h_hist = overall_history.get(home_id, [])
        a_hist = overall_history.get(away_id, [])
        if len(h_hist) < LAST_N or len(a_hist) < LAST_N:
            # Update histories & elo anyway, then continue
            # (no leakage: this match updates after)
            pass
        else:
            # home-only and away-only forms (fallback to overall if insufficient)
            hh_hist = home_history.get(home_id, [])
            aa_hist = away_history.get(away_id, [])

            home_last = last_n_form(h_hist)
            away_last = last_n_form(a_hist)

            if len(hh_hist) >= LAST_N:
                home_home_last = last_n_form(hh_hist)
            else:
                home_home_last = home_last  # fallback

            if len(aa_hist) >= LAST_N:
                away_away_last = last_n_form(aa_hist)
            else:
                away_away_last = away_last  # fallback

            # Elo BEFORE match (no leakage)
            home_elo = elo.get(home_id, ELO_BASE)
            away_elo = elo.get(away_id, ELO_BASE)

            text = build_example_text(
                league_name=league_name,
                season=season,
                home_team=home_team,
                away_team=away_team,
                home_last=home_last,
                away_last=away_last,
                home_home_last=home_home_last,
                away_away_last=away_away_last,
                home_elo=home_elo,
                away_elo=away_elo,
            )

            label = outcome_label(hg, ag)

            examples.append(
                {"date": date, "text": text, "label": label}
            )

        # Now update histories with this match result
        # Home team perspective
        if hg > ag:
            h_res, a_res = "W", "L"
        elif hg == ag:
            h_res, a_res = "D", "D"
        else:
            h_res, a_res = "L", "W"

        overall_history.setdefault(home_id, []).append((h_res, hg, ag))
        overall_history.setdefault(away_id, []).append((a_res, ag, hg))

        home_history.setdefault(home_id, []).append((h_res, hg, ag))
        away_history.setdefault(away_id, []).append((a_res, ag, hg))

        # Update ELO after match
        h_rating = elo.get(home_id, ELO_BASE)
        a_rating = elo.get(away_id, ELO_BASE)
        new_h, new_a = update_elo(h_rating, a_rating, hg, ag)
        elo[home_id] = new_h
        elo[away_id] = new_a

    print(f"Built examples: {len(examples)}")

    # Sort by date to avoid leakage (realistic evaluation)
    examples.sort(key=lambda x: x["date"])

    n = len(examples)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_data = examples[:n_train]
    val_data = examples[n_train:n_train + n_val]
    test_data = examples[n_train + n_val:]

    def dist(data):
        d = {0: 0, 1: 0, 2: 0}
        for x in data:
            d[x["label"]] += 1
        return d

    print("Split:", len(train_data), len(val_data), len(test_data))
    print("Train dist:", dist(train_data))
    print("Val dist:", dist(val_data))
    print("Test dist:", dist(test_data))

    def write_jsonl(path: Path, data: List[dict]):
        with path.open("w", encoding="utf-8") as f:
            for ex in data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    write_jsonl(OUT_DIR / "train.jsonl", train_data)
    write_jsonl(OUT_DIR / "val.jsonl", val_data)
    write_jsonl(OUT_DIR / "test.jsonl", test_data)

    print("Done -> data_out/train.jsonl, val.jsonl, test.jsonl")


if __name__ == "__main__":
    main()
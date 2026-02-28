import json
import math
import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


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
        self.seq = self.seq[-LAST_N:]
        return self

    def extra_stats(self) -> Tuple[int, int, float, float]:
        pts = self.W * 3 + self.D
        gd = self.GF - self.GA
        avg_gf = round(safe_div(self.GF, LAST_N), 3)
        avg_ga = round(safe_div(self.GA, LAST_N), 3)
        return pts, gd, avg_gf, avg_ga


def last_n_form(history: List[Tuple[str, int, int]]) -> TeamForm:
    f = TeamForm()
    for res, gf, ga in history[-LAST_N:]:
        f.add(res, gf, ga)
    return f.finalize()


def form_to_features(f: TeamForm) -> List[float]:
    pts, gd, avg_gf, avg_ga = f.extra_stats()
    # 9 numeric features
    return [
        float(f.W),
        float(f.D),
        float(f.L),
        float(f.GF),
        float(f.GA),
        float(gd),
        float(pts),
        float(avg_gf),
        float(avg_ga),
    ]


def build_compact_text(
    league_name: str,
    season: str,
    home_team: str,
    away_team: str,
    home_elo: float,
    away_elo: float,
    home_last: TeamForm,
    away_last: TeamForm,
    home_home_last: TeamForm,
    away_away_last: TeamForm,
    hh_fallback: int,
    aa_fallback: int,
) -> str:
    # Kompaktan "field" format: manje tokena, vise signala
    # BPE tokenizer ce ovo fino obraditi.
    hp, hgd, havg_gf, havg_ga = home_last.extra_stats()
    ap, agd, aavg_gf, aavg_ga = away_last.extra_stats()

    hhp, hhgd, hhavg_gf, hhavg_ga = home_home_last.extra_stats()
    aap, aagd, aaavg_gf, aaavg_ga = away_away_last.extra_stats()

    return (
        f"LEAGUE={league_name} SEASON={season} "
        f"HOME={home_team} AWAY={away_team} "
        f"HELO={round(home_elo,1)} AELO={round(away_elo,1)} "
        f"H_W={home_last.W} H_D={home_last.D} H_L={home_last.L} H_GF={home_last.GF} H_GA={home_last.GA} H_GD={hgd} H_PTS={hp} H_AVGGF={havg_gf} H_AVGGA={havg_ga} H_SEQ={home_last.seq} "
        f"A_W={away_last.W} A_D={away_last.D} A_L={away_last.L} A_GF={away_last.GF} A_GA={away_last.GA} A_GD={agd} A_PTS={ap} A_AVGGF={aavg_gf} A_AVGGA={aavg_ga} A_SEQ={away_last.seq} "
        f"HH_FALLBACK={hh_fallback} HH_W={home_home_last.W} HH_D={home_home_last.D} HH_L={home_home_last.L} HH_GF={home_home_last.GF} HH_GA={home_home_last.GA} HH_GD={hhgd} HH_PTS={hhp} HH_AVGGF={hhavg_gf} HH_AVGGA={hhavg_ga} HH_SEQ={home_home_last.seq} "
        f"AA_FALLBACK={aa_fallback} AA_W={away_away_last.W} AA_D={away_away_last.D} AA_L={away_away_last.L} AA_GF={away_away_last.GF} AA_GA={away_away_last.GA} AA_GD={aagd} AA_PTS={aap} AA_AVGGF={aaavg_gf} AA_AVGGA={aaavg_ga} AA_SEQ={away_away_last.seq}"
    )


# =========================
# Data extraction
# =========================
def fetch_matches(conn: sqlite3.Connection) -> List[dict]:
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


def main():
    random.seed(SEED)

    if not SQLITE_PATH.exists():
        raise FileNotFoundError(f"Missing {SQLITE_PATH}. Put database.sqlite in data/ folder.")

    conn = sqlite3.connect(str(SQLITE_PATH))
    matches = fetch_matches(conn)
    print(f"Loaded matches: {len(matches)}")

    overall_history: Dict[int, List[Tuple[str, int, int]]] = {}
    home_history: Dict[int, List[Tuple[str, int, int]]] = {}
    away_history: Dict[int, List[Tuple[str, int, int]]] = {}

    elo: Dict[int, float] = {}

    examples = []

    # feature_names (stabilno, uvek isti redosled)
    form_names = ["W", "D", "L", "GF", "GA", "GD", "PTS", "AvgGF", "AvgGA"]
    feature_names = (
        ["home_elo", "away_elo", "elo_diff"]
        + [f"home_last_{n}" for n in form_names]
        + [f"away_last_{n}" for n in form_names]
        + ["hh_fallback"]
        + [f"home_home_last_{n}" for n in form_names]
        + ["aa_fallback"]
        + [f"away_away_last_{n}" for n in form_names]
    )

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

        h_hist = overall_history.get(home_id, [])
        a_hist = overall_history.get(away_id, [])

        if len(h_hist) >= LAST_N and len(a_hist) >= LAST_N:
            hh_hist = home_history.get(home_id, [])
            aa_hist = away_history.get(away_id, [])

            home_last = last_n_form(h_hist)
            away_last = last_n_form(a_hist)

            hh_fallback = 0
            aa_fallback = 0

            if len(hh_hist) >= LAST_N:
                home_home_last = last_n_form(hh_hist)
            else:
                home_home_last = home_last
                hh_fallback = 1

            if len(aa_hist) >= LAST_N:
                away_away_last = last_n_form(aa_hist)
            else:
                away_away_last = away_last
                aa_fallback = 1

            # Elo BEFORE match (no leakage)
            home_elo = float(elo.get(home_id, ELO_BASE))
            away_elo = float(elo.get(away_id, ELO_BASE))
            elo_diff = home_elo - away_elo

            # Compact text (for Transformer)
            text = build_compact_text(
                league_name=league_name,
                season=season,
                home_team=home_team,
                away_team=away_team,
                home_elo=home_elo,
                away_elo=away_elo,
                home_last=home_last,
                away_last=away_last,
                home_home_last=home_home_last,
                away_away_last=away_away_last,
                hh_fallback=hh_fallback,
                aa_fallback=aa_fallback,
            )

            # Numeric features (for MLP)
            features = (
                [home_elo, away_elo, elo_diff]
                + form_to_features(home_last)
                + form_to_features(away_last)
                + [float(hh_fallback)]
                + form_to_features(home_home_last)
                + [float(aa_fallback)]
                + form_to_features(away_away_last)
            )

            label = outcome_label(hg, ag)

            examples.append(
                {
                    "date": date,
                    "text": text,
                    "features": features,
                    "label": label,
                }
            )

        # update histories after using this match
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
        h_rating = float(elo.get(home_id, ELO_BASE))
        a_rating = float(elo.get(away_id, ELO_BASE))
        new_h, new_a = update_elo(h_rating, a_rating, hg, ag)
        elo[home_id] = new_h
        elo[away_id] = new_a

    print(f"Built examples: {len(examples)}")

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
            d[int(x["label"])] += 1
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

    meta = {
        "last_n": LAST_N,
        "num_features": len(feature_names),
        "feature_names": feature_names,
        "elo": {"base": ELO_BASE, "k": ELO_K, "home_adv": ELO_HOME_ADV},
        "splits": {"train_ratio": TRAIN_RATIO, "val_ratio": VAL_RATIO, "test_ratio": 1.0 - TRAIN_RATIO - VAL_RATIO},
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Done -> data_out/train.jsonl, val.jsonl, test.jsonl")
    print("Meta -> data_out/meta.json")


if __name__ == "__main__":
    main()
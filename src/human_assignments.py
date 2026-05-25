from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.human_data import load_human_study

DEFAULT_ASSIGNMENTS_PATH = Path("data/human_mirror_assignments.csv")


def build_mirror_schedule(long_df: pd.DataFrame) -> pd.DataFrame:
    """Build the 85 x 6 participant-to-persona schedule from human study data."""
    pid_order = long_df.drop_duplicates("pid", keep="first")["pid"].tolist()
    pid_to_index = {pid: idx + 1 for idx, pid in enumerate(pid_order)}

    rows: list[dict] = []
    for pid in pid_order:
        participant_rows = long_df[long_df["pid"] == pid]
        for slot_index, (_, row) in enumerate(participant_rows.iterrows(), start=1):
            rows.append(
                {
                    "participant_id": pid,
                    "participant_index": pid_to_index[pid],
                    "slot_index": slot_index,
                    "persona_id": row["persona_id"],
                    "condition": row["condition"],
                }
            )

    schedule = pd.DataFrame(rows)
    if len(schedule) != 510:
        raise ValueError(f"Expected 510 assignments, got {len(schedule)}")
    if schedule.groupby("participant_id").size().nunique() != 1:
        raise ValueError("Participants have unequal assignment counts")
    if not (schedule.groupby("participant_id")["condition"].apply(lambda s: (s == "stereo").sum()) == 3).all():
        raise ValueError("Each participant must have exactly 3 stereotyped personas")

    schedule["evaluation_index"] = range(1, len(schedule) + 1)
    return schedule


def load_mirror_schedule(path: Path | None = None) -> pd.DataFrame:
    assignments_path = path or DEFAULT_ASSIGNMENTS_PATH
    if assignments_path.exists():
        schedule = pd.read_csv(assignments_path)
        required = {
            "participant_id",
            "participant_index",
            "slot_index",
            "persona_id",
            "condition",
            "evaluation_index",
        }
        if not required.issubset(schedule.columns):
            raise ValueError(f"Assignments file missing columns: {required - set(schedule.columns)}")
        return schedule

    long_df, _ = load_human_study()
    schedule = build_mirror_schedule(long_df)
    assignments_path.parent.mkdir(parents=True, exist_ok=True)
    schedule.to_csv(assignments_path, index=False)
    return schedule

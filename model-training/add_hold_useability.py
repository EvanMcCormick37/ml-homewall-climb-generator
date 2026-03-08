"""
add_hold_useability.py

Derives per-hold useability scores from migrated training climbs in storage.db
and pushes the updated hold values to the backend API.

Useability is defined as the inverse of a hold's average appearance difficulty:
holds that tend to appear on easier climbs score higher (more useable), while
holds that only appear on harder climbs score lower.  The raw inverse-mean
values are scaled to [0.01, 1.0] across all holds for a given wall.

Typical usage:
    from add_hold_useability import add_hold_useability

    add_hold_useability(
        api_layout_id="wall-443c15cd12e0",   # backend wall ID
        training_wall_name="decoy",         # wall_id column value in storage.db
    )
"""

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from boardlib_preprocessing import upload_holds

STORAGE_DB   = Path("data/storage.db")
API_BASE_URL = "http://localhost:8000"


def _build_hold_difficulty_df(climbs: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-hold difficulty scores from a DataFrame of migrated climbs.

    For each hold, collects the grade of every climb it appears in, then
    computes inverse-mean difficulty.  This gives holds used on easier climbs
    a higher raw score.  The values are then MinMax-scaled to [0.01, 1.0].

    Args:
        climbs: DataFrame with columns ['holds', 'grade'], where 'holds' is a
                JSON-encoded list of [hold_index, role_int] pairs and 'grade'
                is the numeric difficulty average (non-null).

    Returns:
        DataFrame indexed by hold_index (float) with a single column
        'difficulty_level' (float, range [0.01, 1.0]).
    """
    grades_dict: dict[int, list[float]] = {}

    for _, row in climbs.iterrows():
        try:
            holdset = json.loads(row['holds'])
            for hold_index, role in holdset:
                # We do not need to consider holds which are used as feet
                if role == 3:
                    continue
                if hold_index not in grades_dict:
                    grades_dict[hold_index] = []
                grades_dict[hold_index].append(float(row['grade']))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue

    if not grades_dict:
        raise ValueError("No hold-difficulty data could be extracted from the climbs DataFrame.")

    # Build raw DataFrame: one row per hold, value = 1 / mean(difficulty)
    raw = np.array([[k, 1.0 / np.mean(v)] for k, v in grades_dict.items()])
    df = pd.DataFrame(raw, columns=['hold_index', 'difficulty_level']).set_index('hold_index')

    # Scale inverse-mean values to [0.01, 1.0]  (divide (1,100) range by 100)
    scaler = MinMaxScaler(feature_range=(1, 100))
    df['difficulty_level'] = np.round(
        scaler.fit_transform(df[['difficulty_level']]) / 100.0,
        decimals=2,
    )

    return df


def add_hold_useability(
    api_layout_id: str,
    training_wall_name: str,
    storage_db: str | Path = STORAGE_DB,
    api_base_url: str      = API_BASE_URL,
    min_ascents: int       = 0,
    verbose: bool          = True,
) -> None:
    """
    Compute hold useability scores from storage.db training climbs and push
    the updated holds to the backend API.

    Steps:
        1. Load all climbs for `training_wall_name` from storage.db.
        2. Compute per-hold useability via inverse average difficulty, scaled
           to [0.01, 1.0] across all holds for this wall.
        3. Fetch the current holds list for `api_layout_id` from the backend.
        4. Update the `useability` field on each hold that appears in the
           computed scores; holds with no climb appearances are left unchanged.
        5. PUT the updated holds list back to the backend.

    Args:
        api_layout_id:
            The wall ID used by the backend API (e.g. "wall-443c15cd12e0").
            This is the target of the GET and PUT requests.
        training_wall_name:
            The value of the `wall_id` column in storage.db's climbs table
            for this board (e.g. "decoy", "kilter").  Must match the value
            used during migration (i.e. the boardlib database name).
        storage_db:
            Path to the local storage.db training database.
        api_base_url:
            Base URL for the backend API.
        min_ascents:
            Exclude climbs with fewer than this many ascents from the
            useability calculation.  Defaults to 0 (use all climbs).
            A small threshold (e.g. 5) helps filter out unreliable grades
            from rarely-climbed routes.
        verbose:
            Print progress messages to stdout.
    """
    import requests  # imported here to keep the module importable without requests

    storage_db = Path(storage_db)

    # ------------------------------------------------------------------
    # 1. Load training climbs from storage.db
    # ------------------------------------------------------------------
    if verbose:
        print(f"Loading climbs for wall '{training_wall_name}' from {storage_db} ...")

    query = "SELECT holds, grade FROM climbs WHERE wall_id = ? AND grade IS NOT NULL"
    params: list = [training_wall_name]

    if min_ascents > 0:
        query += " AND ascents >= ?"
        params.append(min_ascents)

    with sqlite3.connect(storage_db) as conn:
        climbs = pd.read_sql_query(query, conn, params=params)

    if climbs.empty:
        raise ValueError(
            f"No climbs found for wall_id='{training_wall_name}' in {storage_db}. "
            "Run db_migration.migrate_boardlib_climbs() first."
        )

    if verbose:
        print(f"  {len(climbs)} climbs loaded.")

    # ------------------------------------------------------------------
    # 2. Compute per-hold useability scores
    # ------------------------------------------------------------------
    if verbose:
        print("Computing hold useability scores ...")

    df_hold_diff = _build_hold_difficulty_df(climbs)

    if verbose:
        print(f"  Scores computed for {len(df_hold_diff)} unique holds.")

    # ------------------------------------------------------------------
    # 3. Fetch current holds from the backend
    # ------------------------------------------------------------------
    if verbose:
        print(f"Fetching holds for API layout '{api_layout_id}' ...")

    endpoint = f"{api_base_url}/api/v1/layouts/{api_layout_id}"
    response = requests.get(endpoint)
    response.raise_for_status()

    holds: list[dict] = response.json()['holds']

    if verbose:
        print(f"  {len(holds)} holds fetched.")

    # ------------------------------------------------------------------
    # 4. Update useability on holds that appear in the training data
    # ------------------------------------------------------------------
    updated = 0
    for hold in holds:
        try:
            score = float(df_hold_diff.loc[float(hold['hold_index']), 'difficulty_level'])
            hold['useability'] = score
            updated += 1
        except KeyError:
            # Hold has no climb appearances — leave its current useability intact
            continue

    if verbose:
        print(f"  {updated}/{len(holds)} holds updated with new useability scores.")

    # ------------------------------------------------------------------
    # 5. PUT updated holds back to the backend
    # ------------------------------------------------------------------
    if verbose:
        print(f"Uploading updated holds to API wall '{api_layout_id}' ...")

    upload_holds(api_layout_id, holds, api_base_url=api_base_url)

    if verbose:
        print("Done.")

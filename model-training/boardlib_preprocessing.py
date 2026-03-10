import collections
import json
from math import isnan
import requests
import sqlite3
import re
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

BOARDLIB_DIR = Path("data/boardlib")
STORAGE_DB   = Path("data/storage.db")

SOURCE_DBS_WITH_LAYOUTS = {
    "kilter":[1,8],
    "tension":[9, 10, 11],
    "grasshopper":[1],
    "decoy":[2]
}

# Maps boardlib placement_role.name → storage.db role integer
ROLE_NAME_TO_INT: dict[str, int] = {
    "start":  0,
    "finish": 1,
    "middle": 2,
    "foot":   3,
}



API_BASE_URL = "http://localhost:8000"

ROLE_MAP = {
    1: 'start',
    2: 'hand',
    3: 'finish',
    4: 'foot'
}
STORAGE_DB   = Path("data/storage.db")

BL_TO_BZ_LAYOUT_ID={
    ('kilter', 1): "layout-0aa86d03949f",
    ('kilter', 8): "layout-95a3f6e2ba1a",
    ('tension', 10): "layout-f712a64fc0dc",
    ('grasshopper', 1): "layout-f6702371e300",
    ('tension', 11): "layout-f311591b6b8b",
    ('decoy', 2): "layout-47aa8b2f3cbc",
    ('tension', 9): "layout-5311e4b5fa08"
}
BZ_TO_BL_LAYOUT_ID = {v:k for k,v in BL_TO_BZ_LAYOUT_ID.items()}

# ---------------------------------------------------------------
#   Hold Upload Utilities
# ---------------------------------------------------------------
def df_sql(db_name, query,index_col=None):
    """Extract table <table_name> from database <db_name>.db"""
    with sqlite3.connect(f"data/boardlib/{db_name}.db") as conn:
        df = pd.read_sql_query(query,conn,index_col=index_col)
    return df

def convert_dataframe_to_holds(
        df: pd.DataFrame,
        index_col: str | None = None,
        x_col: str = 'x_ft',
        y_col: str = 'y_ft',
        default_pull_x: float = 0.0,
        default_pull_y: float = -1.0,
        default_useability: float = 0.5) -> list[dict]:
    """
    Convert pandas DataFrame to list of hold dictionaries.

    Args:
        df: DataFrame with 'hole_id', 'x_ft', 'y_ft' columns
        default_pull_x: Default pull direction x-component
        default_pull_y: Default pull direction y-component
        default_useability: Default difficulty rating (0-1)

    Returns:
        List of hold dictionaries matching the API schema
    """
    holds = []

    for idx, row in df.iterrows():
        hold = {
            "hold_index": int(row[index_col]) if index_col else idx,
            "x": float(row[x_col]),
            "y": float(row[y_col]),
            "pull_x": default_pull_x,
            "pull_y": default_pull_y,
            "useability": default_useability,
        }
        holds.append(hold)

    return holds

def add_useability_features(layout_id, df_hold_diff):
    """Add useability to holds. Requires the holds to be uploaded, and a correctly formatted climbs_df to extract climb difficulty data from."""
    endpoint = f"{API_BASE_URL}/api/v1/layouts/{layout_id}"
    response = requests.get(endpoint,)
    content = json.loads(response.text)
    holds = content['holds']
    for hold in holds:
        try:
            hold['useability'] = float(df_hold_diff.loc[float(hold['hold_index']),'difficulty_level'])
        except:
            continue
    upload_holds(layout_id,holds)

def upload_holds(layout_id: str, holds: list[dict], api_base_url: str = API_BASE_URL):
    """
    Upload holds to the API.
    
    Args:
        layout_id: The wall ID to upload holds to
        holds: List of hold dictionaries
        api_base_url: Base URL of the API
    
    Returns:
        Response JSON from the API
    """
    endpoint = f"{api_base_url}/api/v1/layouts/{layout_id}/holds"
    
    # Prepare form data with JSON-encoded holds
    form_data = {
        "holds": json.dumps(holds)
    }
    
    print(f"Uploading {len(holds)} holds to wall {layout_id}...")
    print(f"Endpoint: {endpoint}")
    
    response = requests.put(endpoint, data=form_data)
    
    if response.status_code == 201:
        print(f"✓ Successfully uploaded {len(holds)} holds!")
        return response.json()
    else:
        print(f"✗ Upload failed with status {response.status_code}")
        print(f"Error: {response.text}")
        response.raise_for_status()

def add_holds(db_name, layout_id_boardlib, layout_id_betazero):
    with sqlite3.connect(f"data/boardlib/{db_name}.db") as conn:
        dfp = pd.read_sql_query("""
        SELECT p.id, h.x, h.y
        FROM (SELECT * FROM placements WHERE layout_id = ?) AS p
        INNER JOIN holes h ON p.hole_id = h.id;
        """, conn, index_col="id", params=(layout_id_boardlib,))
    dfp['x'] -= dfp['x'].min()
    dfp['y'] -= dfp['y'].min()
    dfp['x_ft'] = dfp['x']/12
    dfp['y_ft'] = dfp['y']/12
        
    holds = convert_dataframe_to_holds(dfp)
    upload_holds(layout_id_betazero, holds)

def delete_wall(layout_id: str, api_base_url: str = API_BASE_URL):
    """Remotely delete a wall so I can resize the image."""
    endpoint = f"{api_base_url}/api/v1/layouts/{layout_id}"
    print(f"Deleting wall {layout_id}")
    response = requests.delete(endpoint)
    if response.status_code == 200:
        print("Deletion Successful!")

def parse_frames_to_holdset(frames: str, role_map: dict | None) -> dict:
    """
    Parse frames string to holdset dict.
    
    Format: "p{hold_idx}r{role}p{hold_idx}r{role}..."
    Role mapping: 5=start, 6=hand, 7=finish, 8=foot
    """

    holdset = {
        'start': [],
        'finish': [],
        'hand': [],
        'foot': []
    }

    if role_map is None:
        role_map = ROLE_MAP
    
    # Find all p{int}r{int} patterns
    pattern = r'p(\d+)r(\d+)'
    matches = re.findall(pattern, frames)
    
    for p_str, role_str in matches:
        p = int(p_str)
        role = int(role_str)
        if role in role_map:
            holdset[role_map[role]].append(p)
    
    return holdset

def flush_backup_holds(
    layout_id: str,
    backup_db_path: str = "data/storage.db",
):
    """Function for replacing holds with holds from the back-up DB. Use if you fuck up hold creation."""
    with sqlite3.connect(backup_db_path) as conn:
        df = pd.read_sql_query(
            """
            SELECT hold_index, x, y, pull_x, pull_y, useability, tags
            FROM holds
            WHERE layout_id = ?
            ORDER BY hold_index ASC
            """,conn,
            params=(layout_id,))
    holds = []
    for idx, row in df.iterrows():
        hold = {
            "hold_index": int(row['hold_index']),
            "x": float(row['x']),
            "y": float(row['y']),
            "pull_x": float(row['pull_x']),
            "pull_y": float(row['pull_y']),
            "useability": float(row['useability']),
            "tags": json.loads(row['tags']) if row['tags'] else None
        }
        holds.append(hold)
    upload_holds(layout_id, holds)

def _build_hold_difficulty_dict(climbs: pd.DataFrame, to_hand_ratio = 0.25):
    holds_dict = {}

    for _, row in climbs.iterrows():
        try:
            holdset = json.loads(row['holds'])
            grade = float(row['grade'])
            
            for hold_index, role in holdset:
                if hold_index not in holds_dict:
                    holds_dict[hold_index] = ([],[])
                if role == 3:
                    holds_dict[hold_index][1].append(grade)
                else:
                    holds_dict[hold_index][0].append(grade)
                    
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    
    for k, v in holds_dict.items():
        diffs = v[0] if (len(v[0])>len(v[1])) else v[1]
        is_foot = (len(v[1])*to_hand_ratio > len(v[0]))
        # Instead of using a MinMax scaler, we can follow along with boardlib's default grading scale. 10=V0, 33=V16. To convert to normalized difficulty levels within [0,1],
        # simply subtract a constant and divide by the range, then subtract from 1 to convert from difficulty (higher=harder) to useability (higher=easier).
        useability = round(1 - (np.mean(diffs)-10.0)/23, 2)
        holds_dict[k] = [useability, is_foot]

    return holds_dict

def add_hold_useability(
    api_layout_id: str,
    storage_db: str | Path = STORAGE_DB,
    api_base_url: str      = API_BASE_URL,
    min_ascents: int       = 0,
    to_hand_ratio: float = 0.25,
    verbose: bool          = True,
):
    """
    Compute hold useability scores from storage.db training climbs and push
    the updated holds to the backend API.
    """
    storage_db = Path(storage_db)

    # ------------------------------------------------------------------
    # 1. Fetch current holds from the backend
    # ------------------------------------------------------------------
    if verbose:
        print(f"Fetching holds for API layout '{api_layout_id}' ...")

    endpoint = f"{api_base_url}/api/v1/layouts/{api_layout_id}"
    response = requests.get(endpoint)
    response.raise_for_status()

    holds: list[dict] = response.json()['holds']

    if verbose:
        print(f"  {len(holds)} holds fetched.")

    query = "SELECT holds, grade FROM climbs WHERE layout_id = ? AND grade IS NOT NULL"
    params: list = [api_layout_id]

    if min_ascents > 0:
        query += " AND ascents >= ?"
        params.append(min_ascents)

    with sqlite3.connect(storage_db) as conn:
        climbs = pd.read_sql_query(query, conn, params=params)
    
    if verbose:
        print(f"  {len(climbs)} climbs loaded.")

    if verbose:
        print("Computing hold useability scores ...")


    edit_holds_dict = _build_hold_difficulty_dict(climbs,to_hand_ratio)

    # ------------------------------------------------------------------
    # 3. Update useability on holds that appear in the training data
    # ------------------------------------------------------------------
    updated = 0
    for hold in holds:
        try:
            edit_hold = edit_holds_dict[hold['hold_index']]
            hold['useability'] = edit_hold[0]
            hold['is_foot'] = edit_hold[1]
            updated += 1
        except KeyError:
            continue

    if verbose:
        print(f"  {updated}/{len(holds)} holds updated with new useability scores.")

    # ------------------------------------------------------------------
    # 4. PUT updated holds back to the backend
    # ------------------------------------------------------------------
    if verbose:
        print(f"Uploading updated holds to API wall '{api_layout_id}' ...")

    upload_holds(api_layout_id, holds, api_base_url=api_base_url)

    if verbose:
        print("Done.")
def add_default_hold_roles(
    bz_layout_id: str,
):
    """Update BZ holds list with BL default hold roles"""
    endpoint = f"http://localhost:8000/api/v1/layouts/{bz_layout_id}"
    response = requests.get(endpoint)
    response.raise_for_status()
    holds = response.json()['holds']
    holds_dict = {hold['hold_index']:hold for hold in holds}

    (db_name, bl_layout_id) = BZ_TO_BL_LAYOUT_ID[bz_layout_id]

    with sqlite3.connect(f"data/boardlib/{db_name}.db") as conn:
        conn.row_factory = sqlite3.Row
        role_map = _build_role_map(conn)
        rows = conn.execute(
            f"SELECT id, layout_id, default_placement_role_id FROM placements WHERE layout_id = ?",
            (bl_layout_id,)
        ).fetchall()
        for row in rows:
            try:
                is_foot = (role_map[row['default_placement_role_id']] == 3)
                holds_dict[row['id']]['is_foot'] = is_foot
            except KeyError:
                continue
    holds_list = list(holds_dict.values())
    upload_holds(bz_layout_id, holds_list)
    
    with sqlite3.connect("data/storage.db") as conn:
        batch = [(
            f"{bz_layout_id}-{hold['hold_index']}",
            bz_layout_id,
            hold['hold_index'],
            hold['x'],
            hold['y'],
            hold['pull_x'],
            hold['pull_y'],
            hold['useability'],
            hold['is_foot'],
            json.dumps(hold['tags']),
            datetime.now(),
            datetime.now(),
        ) for hold in holds_list]
        conn.executemany(
            """
            INSERT OR IGNORE INTO holds (id, layout_id, hold_index, x, y,
            pull_x, pull_y, useability, is_foot, tags, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, batch)
        conn.commit()
    print(f"Successfully uploaded {len(holds_dict)} holds to training DB!")
def move_holds_to_db(
    bz_layout_id
):
    endpoint = f"http://localhost:8000/api/v1/layouts/{bz_layout_id}"
    response = requests.get(endpoint)
    response.raise_for_status()
    holds = response.json()['holds']
    with sqlite3.connect("data/storage.db") as conn:
        conn.execute("DELETE FROM holds WHERE layout_id = ?",(bz_layout_id,))
        batch = [(
            f"{bz_layout_id}-{hold['hold_index']}",
            bz_layout_id,
            hold['hold_index'],
            hold['x'],
            hold['y'],
            hold['pull_x'],
            hold['pull_y'],
            hold['useability'],
            hold['is_foot'],
            json.dumps(hold['tags']),
            datetime.now(),
            datetime.now(),
        ) for hold in holds]
        conn.executemany(
            """
            INSERT OR IGNORE INTO holds (id, layout_id, hold_index, x, y,
            pull_x, pull_y, useability, is_foot, tags, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, batch)
        conn.commit()
    print(f"Successfully uploaded {len(holds)} holds to training DB!")
# ---------------------------------------------------------------
#   Climb Upload Utilities
# ---------------------------------------------------------------
def upload_climbs_batch(
    layout_id: str,
    df: pd.DataFrame,
    role_map: dict | None = None,
    base_url: str = API_BASE_URL,
    batch_size: int = 1000,
    verbose: bool = True,
    try_one: bool = False,
) -> list[dict]:
    """
    Upload climbs from DataFrame to backend using batch endpoint.
    
    Args:
        df: DataFrame with columns [uuid, angle, frames, difficulty_average, 
            quality_average, ascensionist_count, fa_username, created_at]
        layout_id: The wall ID to upload climbs to
        base_url: Base URL of the API
        batch_size: Number of climbs per batch request
        verbose: Print progress updates
        try_one: If True, only process the first row (for testing)
    
    Returns:
        List of results with original uuid and response/error
    """
    endpoint = f"{base_url}/api/v1/layouts/{layout_id}/climbs/batch"
    results = []
    
    if try_one:
        df = df.head(1)
    
    total = len(df)
    
    # Process in batches
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_df = df.iloc[batch_start:batch_end]
        
        # Build payloads for this batch, tracking original uuids
        climbs = []
        batch_uuids = []
        
        for idx, row in batch_df.iterrows():
            try:
                # Parse frames to holdset
                holdset = parse_frames_to_holdset(row['frames'], role_map)
                
                # Build payload (same logic as original)
                payload = {
                    'name': f"{row['name']}-{row['angle']}",
                    'holdset': holdset,
                    'angle': int(row['angle']),
                    'grade': float(row['difficulty_average']),
                    'quality': float(row['quality_average']),
                    'ascents': int(row['ascensionist_count']),
                    'setter_name': row.get('fa_username') if pd.notna(row.get('fa_username')) else None,
                    'tags': None
                }
                
                climbs.append(payload)
                batch_uuids.append(row['name'])
                
            except Exception as e:
                # If payload construction fails, record error immediately
                results.append({
                    'original_uuid': row['name'],
                    'status': 'error',
                    'error': f"Payload construction failed: {str(e)}"
                })
        
        # Send batch request if we have climbs
        if climbs:
            try:
                response = requests.post(endpoint, json={'climbs': climbs})
                
                if response.status_code == 201:
                    batch_results = response.json()['results']
                    
                    # Map batch results back to per-row format
                    for result in batch_results:
                        idx = result['index']
                        if result['status'] == 'success':
                            results.append({
                                'original_uuid': batch_uuids[idx],
                                'new_id': result['id'],
                                'status': 'success'
                            })
                        else:
                            results.append({
                                'original_uuid': batch_uuids[idx],
                                'status': 'error',
                                'error': result.get('error', 'Unknown error')
                            })
                else:
                    # Entire batch request failed
                    for uuid in batch_uuids:
                        results.append({
                            'original_uuid': uuid,
                            'status': 'error',
                            'error': f"Batch request failed: {response.text}"
                        })
                        
            except Exception as e:
                # Network/request error - mark all in batch as failed
                for uuid in batch_uuids:
                    results.append({
                        'original_uuid': uuid,
                        'status': 'error',
                        'error': f"Request failed: {str(e)}"
                    })
        
        # Progress update
        if verbose:
            success_count = sum(1 for r in results if r['status'] == 'success')
            print(f"Progress: {len(results)}/{total} | Success: {success_count} | Errors: {len(results) - success_count}")
    
    return results

# ---------------------------------------------------------------------------
# Climb DB Migration
# ---------------------------------------------------------------------------

def _build_role_map(src_conn: sqlite3.Connection) -> dict[int, int]:
    """
    Build a mapping from placement_role.id (int) to storage.db role integer
    by reading the source database's placement_roles table.

    boardlib stores role names ('start', 'finish', 'middle', 'foot') in
    placement_roles.name; the IDs vary across board types (e.g. kilter uses
    IDs 12-15, 20-23, …, while tension/grasshopper/decoy use 1-4).

    Args:
        src_conn: Open connection to a boardlib source database.

    Returns:
        Dict mapping each known role ID to its storage.db integer (0-3).
        IDs whose name is not in ROLE_NAME_TO_INT (e.g. kilter's 'cyan',
        'magenta' coloured-hold roles) are omitted — holds with those role
        IDs will be dropped during frame parsing.
    """
    rows = src_conn.execute("SELECT id, name FROM placement_roles").fetchall()
    role_map: dict[int, int] = {}
    for role_id, name in rows:
        role_int = ROLE_NAME_TO_INT.get(name)
        if role_int is not None:
            role_map[role_id] = role_int
    return role_map


def _parse_frames(frames: str, role_map: dict[int, int]) -> list[list[int]]:
    """
    Parse a boardlib frames string into a list of [hold_index, role_int] pairs.

    The frames format is a concatenation of tokens: p{hole_id}r{role_id}
    e.g. "p1100r12p1200r13p1350r14p1400r15"

    Args:
        frames:   Raw frames string from boardlib climbs.frames.
        role_map: Mapping from boardlib role ID to storage.db role integer,
                  as returned by _build_role_map().

    Returns:
        List of [hold_index, role_int] pairs in parse order.
        Holds whose role_id is absent from role_map are silently dropped
        (this covers kilter's decorative colour roles).
        Returns an empty list if the string is unparseable or all roles unknown.
    """
    holds: list[list[int]] = []
    for p_str, r_str in re.findall(r'p(\d+)r(\d+)', frames):
        role_int = role_map.get(int(r_str))
        if role_int is not None:
            holds.append([int(p_str), role_int])
    return holds

def migrate_boardlib_climbs(
    source_db_names: dict[str,list[int]] = SOURCE_DBS_WITH_LAYOUTS,
    boardlib_dir: str | Path    = BOARDLIB_DIR,
    storage_db:   str | Path    = STORAGE_DB,
    min_ascents:  int           = 2,
    verbose:      bool          = True,
) -> dict[str, int]:
    """
    Migrate climbs from boardlib SQLite databases into data/storage.db.

    For each source database, every row in climb_stats is treated as one
    training example — this naturally handles boards (kilter, tension) where
    a single climb can be set at multiple angles, each with independent grade
    and quality statistics.  The join is:

        climb_stats  INNER JOIN  climbs  ON  uuid = climb_uuid

    so only climbs that have at least one stats row are migrated.  Climbs
    without stats carry no grade or quality signal and are not useful for
    conditional generation training.

    Schema mapping (boardlib → storage.db climbs):
        id          ← climbs.uuid  (used as a stable source identifier)
        layout_id     ← <db_name>-<layout_id>  (e.g. "kilter-1"; no matching row in walls table)
        angle       ← climb_stats.angle
        holds       ← JSON [[hold_index, role_int], ...]  parsed from climbs.frames
        grade       ← climb_stats.difficulty_average
        quality     ← climb_stats.quality_average
        ascents     ← climb_stats.ascensionist_count

    Args:
        source_db_names:
            List of boardlib database names to process, without the .db
            extension.  Defaults to ["kilter", "tension", "grasshopper", "decoy"].
        boardlib_dir:
            Directory containing the boardlib .db files.
        storage_db:
            Path to the destination storage.db file.
        min_ascents:
            Skip climb-angle pairs with fewer than this many ascents.
            Defaults to 0 (keep everything).  A value of ~10 is a reasonable
            filter to remove rarely-climbed and potentially mislabelled routes.
        skip_existing:
            If True (default), check for rows already present in storage.db
            (matched by the synthetic id column) and skip them.  Safe to re-run
            incrementally after a partial migration.
        verbose:
            Print per-database progress updates to stdout.

    Returns:
        Dict of {db_name: rows_inserted} for each processed source database.
    """
    boardlib_dir = Path(boardlib_dir)
    storage_db   = Path(storage_db)

    summary: dict[str, int] = {}

    with sqlite3.connect(storage_db) as dest:

        for db_name, layouts in source_db_names.items():
            src_path = boardlib_dir / f"{db_name}.db"

            if not src_path.exists():
                if verbose:
                    print(f"[{db_name}] {src_path} not found — skipping.")
                summary[db_name] = 0
                continue

            if verbose:
                print(f"\n[{db_name}] Opening {src_path} ...")

            inserted = 0
            skipped  = 0

            for layout in layouts:
                with sqlite3.connect(src_path) as src:
                    src.row_factory = sqlite3.Row

                    # Build role_id → role_int map for this board type
                    role_map = _build_role_map(src)
                    if verbose:
                        print(f"[{db_name}] Role map: {role_map}")

                    min_ascents_clause = (
                        f"AND cs.ascensionist_count >= {int(min_ascents)}"
                        if min_ascents > 0 else ""
                    )
                    query = f"""
                        SELECT
                            c.uuid            AS uuid,
                            c.frames          AS frames,
                            cs.angle          AS angle,
                            cs.difficulty_average  AS grade,
                            cs.quality_average     AS quality,
                            cs.ascensionist_count  AS ascents
                        FROM climb_stats cs
                        INNER JOIN climbs c ON c.uuid = cs.climb_uuid
                        WHERE c.layout_id = ?
                            AND c.is_listed = 1
                            AND c.is_draft  = 0
                            AND c.frames IS NOT NULL
                            AND c.frames   != ''
                        {min_ascents_clause}
                    """

                    rows = src.execute(query, (layout,)).fetchall()

                if verbose:
                    print(f"[{db_name}, layout id {layout}] {len(rows)} candidate (climb, angle) pairs.")

                batch: list[tuple] = []

                for row in rows:

                    holds = _parse_frames(row['frames'], role_map)
                    layout_id = BL_TO_BZ_LAYOUT_ID[(db_name,layout)]
                    if not holds:
                        # All frame tokens had unknown role IDs (e.g. purely decorative)
                        skipped += 1
                        continue


                    batch.append((
                        row['uuid'],            # id
                        layout_id,              # layout_id
                        row['angle'],           # angle
                        json.dumps(holds),      # holds
                        row['grade'],           # grade
                        row['quality'],         # quality
                        row['ascents'],         # ascents
                    ))

                    # Flush in batches of 1 000 to keep memory usage flat
                    if len(batch) == 1_000:
                        dest.executemany(
                            """
                            INSERT OR IGNORE INTO climbs
                            (id, layout_id, angle, holds,
                            grade, quality, ascents)
                            VALUES (?,?,?,?,?,?,?)
                            """,
                            batch,
                        )
                        inserted += len(batch)
                        batch = []
                        if verbose:
                            print(f"[{db_name}]   {inserted} inserted so far ...")

                # Final partial batch
                if batch:
                    dest.executemany(
                        """
                        INSERT OR IGNORE INTO climbs
                            (id, layout_id, angle, holds,
                            grade, quality, ascents)
                        VALUES (?,?,?,?,?,?,?)
                        """,
                        batch,
                    )
                    inserted += len(batch)

                dest.commit()
                summary[db_name] = inserted

                if verbose:
                    print(
                        f"[{db_name}] Done — {inserted} inserted, {skipped} skipped."
                    )

    if verbose:
        total = sum(summary.values())
        print(f"\nMigration complete. Total rows inserted: {total}")
        for name, count in summary.items():
            print(f"  {name:>12}: {count:>8,} rows")

    return summary

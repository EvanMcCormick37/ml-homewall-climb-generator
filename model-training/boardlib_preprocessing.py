import json
import requests
import pandas as pd
import sqlite3
import re

API_BASE_URL = "http://localhost:8000"

ROLE_MAP = {
    1: 'start',
    2: 'hand',
    3: 'finish',
    4: 'foot'
}

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

# ---------------------------------------------------------------
#   Climb Upload Utilities
# ---------------------------------------------------------------
def upload_climbs_batch(
    layout_id: str,
    df: pd.DataFrame,
    role_map: str | None = None,
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
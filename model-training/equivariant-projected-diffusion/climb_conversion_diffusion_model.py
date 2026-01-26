import numpy as np
import pandas as pd
import sqlite3
import json
import math

def get_preprocessed_3d_features(db_path: str, wall_id_pattern: str) -> np.ndarray:
    """
    Loads climb/hold data, filters outliers, pads sequences, and applies 
    geometric transformations to return a 3D feature array.
    """
    
    # --- Helper: 3D Geometry Transform ---
    def apply_wall_angle(angle, val):
        rads = math.radians(angle)
        return val * math.cos(rads), val * math.sin(rads)

    # --- Step 1: Data Loading & Preprocessing ---
    with sqlite3.connect(db_path) as conn:
        # Load Climbs
        fdf = pd.read_sql_query(
            "SELECT id, angle, holds FROM climbs WHERE ascents > 1", 
            conn, index_col="id"
        )
        fdf['holds'] = fdf['holds'].apply(json.loads)
        fdf = fdf[fdf['holds'].str.len() <= 20].copy()
        fdf['angle'] = fdf['angle'].astype(int)

        # Load Holds
        hdf = pd.read_sql_query(
            "SELECT hold_index, x, y, pull_x, pull_y, useability, is_foot FROM holds WHERE wall_id LIKE ?", 
            conn, params=(wall_id_pattern,), index_col='hold_index'
        )
        # Apply usability factor
        hdf['pull_y'] *= hdf['useability']
        hdf['pull_x'] *= hdf['useability']
        hdf.drop(columns='useability', inplace=True)

    # --- Step 2: Feature Construction Loop ---
    features_3d = []
    target_length = 30

    for _, row in fdf.iterrows():
        try:
            # Pad sequence with random NULL holds (role 4)
            pad_len = target_length - len(row['holds'])
            padding = list(zip(np.random.choice(hdf.index, size=pad_len), [4] * pad_len))
            padded_list = row['holds'] + padding

            climb_features = []
            for h_idx, role in padded_list:
                # Extract features
                x, y, px, py, is_foot = hdf.loc[h_idx]
                
                # Apply 3D transformations based on wall angle
                y_3d, z_3d = apply_wall_angle(row['angle'], y)
                py_3d, pz_3d = apply_wall_angle(row['angle'], py)
                
                climb_features.append([x, y_3d, z_3d, px, py_3d, pz_3d, is_foot, role])
            
            features_3d.append(climb_features)
        except KeyError:
            continue # Skip climbs referencing missing holds

    return np.array(features_3d, dtype=np.float32)
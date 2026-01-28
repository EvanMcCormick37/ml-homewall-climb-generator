import numpy as np
import pandas as pd
import sqlite3
import json
import math
import torch
from dataclasses import dataclass
from torch_geometric.data import Data, InMemoryDataset
from sklearn.preprocessing import StandardScaler

@dataclass
class PreprocessingOptions:
    min_ascents: int = 1
    pad_length: int = 20

@dataclass
class FeatureIndices:
    pos: list[int]
    vec: list[int]
    scalars: list[int]
    role: int

def apply_wall_angle(angle, val):
    rads = math.radians(angle)
    return val * math.cos(rads), val * math.sin(rads)

class ClimbingDataset(InMemoryDataset):
    def __init__(self, features_array: np.ndarray, indices: FeatureIndices):
        super().__init__(root=None, transform=None, pre_transform=None)
        
        data_list = []
        for climb in features_array:
            tensor = torch.from_numpy(climb).float()
            
            # Map columns to ClimbBatch protocol expectations
            pos = tensor[:, indices.pos]       # x, y, z
            vec = tensor[:, indices.vec]       # pull_x, pull_y, pull_z
            
            # SCALARS: Currently only 'is_foot' (index 6). 
            scalars = tensor[:, indices.scalars]   
            
            # ROLES: Index 7. Must be Long type for Embedding layers
            roles = tensor[:, indices.role].long() 

            # Create PyG Data object
            data = Data(
                pos=pos,
                vec=vec,
                scalars=scalars,
                roles=roles,
                num_nodes=features_array.shape[1]
            )
            data_list.append(data)
            
        self.data, self.slices = self.collate(data_list)

def get_preprocessed_3d_features(db_path: str, wall_id: str, options: PreprocessingOptions = PreprocessingOptions()) -> ClimbingDataset:
    """
    Loads climb/hold data, filters outliers, pads sequences, and applies 
    geometric transformations to return a 3D feature array.
    """

    # --- Step 1: Data Loading & Preprocessing ---
    with sqlite3.connect(db_path) as conn:
        # Load Climbs
        fdf = pd.read_sql_query(
            f"SELECT id, angle, holds FROM climbs WHERE ascents > {options.min_ascents}", 
            conn, index_col="id"
        )
        fdf['holds'] = fdf['holds'].apply(json.loads)
        fdf = fdf[fdf['holds'].str.len() <= min(20, options.pad_length)].copy()
        fdf['angle'] = fdf['angle'].astype(int)

        # Load Holds
        hdf = pd.read_sql_query(
            "SELECT hold_index, x, y, pull_x, pull_y, useability, is_foot FROM holds WHERE wall_id LIKE ?", 
            conn, params=(wall_id,), index_col='hold_index'
        )
        # Apply usability factor
        hdf['pull_y'] *= hdf['useability']
        hdf['pull_x'] *= hdf['useability']
        hdf.drop(columns='useability', inplace=True)

    # --- Step 2: Feature Construction Loop ---
    features_3d = []

    for _, row in fdf.iterrows():
        try:
            # Pad sequence with random NULL holds (role 4)
            pad_len = options.pad_length - len(row['holds'])
            padding = list(zip(np.random.choice(hdf.index, size=pad_len), [4] * pad_len))
            padded_list = row['holds'] + padding

            climb_features = []
            for h_idx, role in padded_list:
                # Extract features
                x, y, px, py, is_foot = hdf.loc[h_idx]
                
                # Apply 3D transformations based on wall angle
                y_3d, z_3d = apply_wall_angle(row['angle'], y)
                py_3d, pz_3d = apply_wall_angle(row['angle'], py)
                
                # Normalize spatial features to [0,1] by dividing by max_feet (15 ft)
                climb_features.append([x/15.0, y_3d/15.0, z_3d/15.0, px, py_3d, pz_3d, is_foot, role])
            
            features_3d.append(climb_features)
        except KeyError:
            continue # Skip climbs referencing missing holds
    
    feature_array = np.array(features_3d, dtype=np.float32)
    indices = FeatureIndices(
        pos = [0,1,2],
        vec=[3,4,5],
        scalars=[6],
        role=7
    )
    return ClimbingDataset(feature_array, indices)
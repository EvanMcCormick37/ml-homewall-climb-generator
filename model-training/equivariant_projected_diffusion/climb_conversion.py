import numpy as np
import pandas as pd
import sqlite3
import json
import math
import torch
from dataclasses import dataclass
from torch_geometric.data import Data, InMemoryDataset
from sklearn.preprocessing import MinMaxScaler

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

def zero_start_hold_position(f_arr, dim=2):
    """Return the 3d position of the starting holds (or the mean 3d position if there are two)"""
    start_holds = f_arr[f_arr[:,dim*2+1]==1.0]
    start_pos = np.mean(start_holds[:,:dim], axis=0)
    f_arr[:,:dim] -= start_pos
    
    return f_arr

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


class ClimbsFeatureArray:
    def __init__(
            self,
            db_path: str = "data/storage.db",
            to_length: int = 20
        ):
        """Gets climbs from the climbs database and converts them into featured Sequence data for our DDPM."""
        self.scaler = MinMaxScaler(feature_range=(-1,1))
        with sqlite3.connect(db_path) as conn:
                # Load Climbs
                fdf = pd.read_sql_query(
                    f"SELECT * FROM climbs WHERE ascents > 1", 
                    conn, index_col="id"
                )
                fdf['holds'] = fdf['holds'].apply(json.loads)
                fdf = fdf[fdf['holds'].str.len() <= to_length].copy()
                fdf['angle'] = fdf['angle'].astype(int)
                # Normalize these columns for feature conditioning. Mean = 0, Std. Dev = 1
                fdf['grade'] -= fdf['grade'].mean()
                fdf['quality'] -= fdf['quality'].mean()
                fdf['ascents'] -= fdf['ascents'].mean()
                fdf['grade'] = self.scaler.fit_transform(fdf[['grade']])
                fdf['quality'] = self.scaler.fit_transform(fdf[['quality']])
                fdf['ascents'] = self.scaler.fit_transform(fdf[['ascents']])

                # Load Holds
                hdf = pd.read_sql_query(
                    "SELECT hold_index, x, y, pull_x, pull_y, useability, is_foot FROM holds WHERE wall_id LIKE ?", 
                    conn, params=('wall-443c15cd12e0',), index_col='hold_index'
                )
                # Apply usability factor
                hdf['pull_y'] *= hdf['useability']
                hdf['pull_y'] = -hdf['pull_y']
                hdf['pull_x'] *= hdf['useability']
                # Scale climbs to unit scale They will eventually be in range (-1,1), so divide by 1/2 wall size (12x12).
                hdf['x'] /= 6
                hdf['y'] /= 6
                hdf.drop(columns='useability', inplace=True)
        
        self.holds_df = hdf
        self.climbs_df = fdf
        self.to_length = to_length
        self.db_path = db_path
    
    def get_features_3d(self):
        """Converts a list of hold indices + roles to a 3d-point cloud and pads the sequence by adding NULL holds."""
        x_features = []
        cond_features = []
        for _, row in self.climbs_df.iterrows():
            try:
                x, cond = self.embed_features_3d(row, self.to_length, self.holds_df)
                x_features.append(x)
                cond_features.append(cond)
            except:
                continue
        return np.array(x_features), np.array(cond_features)
    
    def get_features_2d(self):
        """Converts a list of hold indices + roles to a 3d-point cloud and pads the sequence by adding NULL holds."""
        x_features = []
        cond_features = []
        for _, row in self.climbs_df.iterrows():
            try:
                x, cond = self.embed_features_2d(row, self.to_length, self.holds_df)
                x_features.append(x)
                cond_features.append(cond)
            except:
                continue
        return np.array(x_features), np.array(cond_features)

        

    def embed_features_2d(self, row, to_length, hdf):
        """Converts a list of hold indices + roles to a 2d-point cloud and pads the sequence by adding random NULL holds."""
        pad_length = to_length-len(row["holds"])
        assert pad_length >= 0

        _2d_features = []
        eye_encoder = np.eye(5)
        # Initial embeddings.
        for hold in row["holds"]:
            # One-hot encoded role: [0: Start, 1: Finish, 2: Generic Hand, 3: Generic Foot, 4: NULL]
            role_encoding = list(eye_encoder[hold[1]])
            # print(role_encoding)
            # Get hold features from hold df[hold_idx].
            features = hdf.loc[hold[0]]
            x, y, pull_x, pull_y, is_foot = [f for f in features]
            features=[x, y, pull_x, pull_y, is_foot]+[ohr for ohr in role_encoding]
            _2d_features.append(features)
        features_arr = np.array(_2d_features, dtype=np.float32)
        # Zero-centering climb around the starting holds.
        features_arr = zero_start_hold_position(features_arr)
        # Re-order holds based on distance from starting holds.
        features_arr = np.array(sorted(features_arr, key=(lambda arr: np.sqrt(arr[0]**2+arr[1]**2+arr[2]**2))))
        # Pad with NULL holds:
        null_holds = np.concatenate([np.full((pad_length,features_arr.shape[1]-1),0),np.ones((pad_length,1))], axis=1)
        hold_features = np.concatenate([features_arr,null_holds], axis=0, dtype=np.float32)
        
        # Add conditioning features (Grade, Rating, Ascents)
        conditional_features = np.array([row['grade'], row['quality'],row['ascents'],row['angle']/90], dtype=np.float32)
        return (hold_features, conditional_features)
    
    def embed_features_3d(self, row, to_length, hdf):
        """Embed a single row of the climbs dataframe into 3d features."""
        pad_length = to_length-len(row["holds"])
        assert pad_length >= 0

        _3d_features = []
        eye_encoder = np.eye(5)
        # Initial embeddings.
        for hold in row["holds"]:
            # One-hot encoded role: [0: Start, 1: Finish, 2: Generic Hand, 3: Generic Foot, 4: NULL]
            role_encoding = list(eye_encoder[hold[1]])
            # print(role_encoding)
            # Get hold features from hold df[hold_idx].
            features = self.holds_df.loc[hold[0]]
            x, y, pull_x, pull_y, is_foot = [f for f in features]
            # print(x, y, pull_x, pull_y, is_foot)
            y, z = apply_wall_angle(row["angle"], y)
            pull_y, pull_z = apply_wall_angle(row["angle"], pull_y)
            features=[x, y, z, pull_x, pull_y, pull_z, is_foot]+[ohr for ohr in role_encoding]
            _3d_features.append(features)
        features_arr = np.array(_3d_features, dtype=np.float32)
        # Zero-centering climb around the starting holds.
        features_arr = zero_start_hold_position(features_arr,dim=3)
        # Re-order holds based on distance from starting holds.
        features_arr = np.array(sorted(features_arr, key=(lambda arr: np.sqrt(arr[0]**2+arr[1]**2+arr[2]**2))))
        # Pad with NULL holds:
        null_holds = np.concatenate([np.full((pad_length,features_arr.shape[1]-1),0),np.ones((pad_length,1))], axis=1)
        hold_features = np.concatenate([features_arr,null_holds], axis=0, dtype=np.float32)
        
        # Add conditioning features (Grade, Rating, Ascents)
        conditional_features = np.array([row['grade'], row['quality'],row['ascents'],row['angle']/90], dtype=np.float32)
        return (hold_features, conditional_features)
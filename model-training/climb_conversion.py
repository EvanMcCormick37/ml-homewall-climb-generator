import sqlite3
import json
import math
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import TensorDataset
import joblib

SCALER_WEIGHTS_PATH = 'data/weights/climbs-feature-scaler.joblib'
DDPM_WEIGHTS_PATH = 'data/weights/simpler-diffusion-large.pth'
DB_PATH = 'data/storage.db'

class ClimbsFeatureArray:
    def __init__(
            self,
            db_path: str = DB_PATH,
            to_length: int = 20,
            load_weights_path: str | None = SCALER_WEIGHTS_PATH,
            save_weights_path: str | None = SCALER_WEIGHTS_PATH
        ):
        """Gets climbs from the climbs database and converts them into featured Sequence data."""
        print("Initializing ClimbsFeatureArray...")
        self.db_path = db_path
        self.to_length = to_length
        self.scaler = ClimbsFeatureScaler(weights_path=load_weights_path)

        self.null_token = [-2,0,-2,0,0,0,0,0,1] 

        with sqlite3.connect(db_path) as conn:
            # Load climbs
            climbs_to_fit = pd.read_sql_query("SELECT * FROM climbs WHERE ascents > 1", conn, index_col='id')
            climbs_to_fit['holds'] = climbs_to_fit['holds'].apply(json.loads)
            # Filter by length
            climbs_to_fit = climbs_to_fit[climbs_to_fit['holds'].map(len) <= to_length]

            # Load holds
            holds_to_fit = pd.read_sql_query("SELECT hold_index, x, y, pull_x, pull_y, useability, is_foot, wall_id FROM holds", conn)
            
            # Scale
            scaled_climbs, scaled_holds = self.scaler.transform(climbs_to_fit, holds_to_fit)
            self.hold_features_df = scaled_holds
            
            # Create fast lookup
            self.holds_lookup = {
                wall_id: group.drop(columns=['wall_id','useability', 'is_foot', 'mult']).set_index('hold_index').to_dict('index')
                for wall_id, group in scaled_holds.groupby('wall_id')
            }
            self.climbs_df = scaled_climbs
        
        if save_weights_path is not None:
            self.scaler.save_weights(save_weights_path)
        
        print(f"ClimbsFeatureArray initialized! {len(self.climbs_df)} unique climbs added!")

    # --- Feature Engineering Helpers ---
    def apply_wall_angle(self, angle, val):
        rads = math.radians(angle)
        return val * math.cos(rads), val * math.sin(rads)

    def _zero_center_of_mass(self, arr, dim=2, start_holds: list | None = None):
        """Center the climb on the mean position of the start holds (or all holds)."""
        if start_holds is not None and len(start_holds) > 0:
            com = np.mean(np.array(start_holds)[:,:dim], axis=0)
        else:
            com = np.mean(arr[:,:dim], axis=0)
        arr[:,:dim] -= com
        return arr

    def _2d_features(self, h_data):
        return [h_data['x'], h_data['y'], h_data['pull_x'], h_data['pull_y']]
    
    def _3d_features(self, h_data, angle):
        y, z = self.apply_wall_angle(h_data['y'], angle)
        py, pz = self.apply_wall_angle(h_data['pull_y'], angle)
        return [h_data['x'], y, z, h_data['pull_x'], py, pz]

    # --- Core Extraction Logic ---
    def _extract_sequences(self, dim: int, limit: int | None, roles: bool):
        """
        Iterates through the dataframe and extracts raw lists of features, conditions, targets, and lengths.
        Returns: (x_list, cond_list, targets_list, lengths_list)
        """
        x_out, cond_out, targets_out, lengths_out = [], [], [], []

        df = self.climbs_df.copy()
        if limit is not None:
            df = df.iloc[:limit,:]

        for _, row in df.iterrows():
            try:
                # 1. Fetch Data
                climb_holds = row['holds']
                wall_id = row['wall_id']
                angle = row['angle']
                wall_holds = self.holds_lookup[wall_id]
                
                start_holds, middle_holds, finish_holds = [], [], []
                
                # 2. Process Holds
                for h_idx, role in climb_holds:
                    h_data = wall_holds[h_idx]
                    
                    if dim == 3:
                        feat = self._3d_features(h_data, angle)
                    else:
                        feat = self._2d_features(h_data)
                    
                    if roles:
                        role_vec = [0.0]*5
                        role_vec[role] = 1.0
                        feat.extend(role_vec)
                    
                    match role:
                        case 0: start_holds.append(feat)
                        case 1: finish_holds.append(feat)
                        case _: middle_holds.append(feat)

                # 4. Sort & Flatten
                # Sorting by Y (vertical) or X+Y helps the model learn sequential patterns
                sort_key = lambda x: x[0] + x[1]
                f_arr = np.array([
                    h for hl in [start_holds, middle_holds, finish_holds]
                    for h in sorted(hl, key=sort_key)
                ])

                # 5. Center & Pad
                f_arr = self._zero_center_of_mass(f_arr, dim=dim, start_holds=start_holds)
                actual_length = len(f_arr)
                pad_length = self.to_length - actual_length

                if pad_length > 0:
                    null_feat_len = 9 if roles else 4
                    curr_null = self.null_token[:null_feat_len]
                    null_holds = np.tile(np.array(curr_null), (pad_length, 1))
                    f_arr = np.concatenate([f_arr, null_holds], axis=0, dtype=np.float32)
                elif pad_length < 0:
                    continue

                # 6. Append
                x_out.append(f_arr)
                cond_out.append([row['grade'], row['quality'], row['ascents'], angle])

            except KeyError:
                continue

        return x_out, cond_out

    def _augment_and_tensorize(self, x_list, cond_list, augment=True, dim=2):
        """
        Converts lists to Numpy, performs reflection augmentation, and returns Numpy arrays.
        """
        x_arr = np.array(x_list, dtype=np.float32)
        cond_arr = np.array(cond_list, dtype=np.float32)

        if augment:
            refl_x = x_arr.copy()
            
            refl_x[:, :, [0, dim]] *= -1
            
            x_arr = np.concatenate([x_arr, refl_x], axis=0)
            cond_arr = np.tile(cond_arr, (2, 1))

        return x_arr, cond_arr

    def get_features(self, dim=2, limit: int | None = None, roles = False, augment_reflections=True):
        """
        Original method for Diffusion models.
        Returns: TensorDataset(x_seq, cond)
        """
        # 1. Extract
        x, c = self._extract_sequences(dim, limit, roles)
        
        # 2. Augment
        x_arr, c_arr = self._augment_and_tensorize(x, c, augment=augment_reflections, dim=dim)
        
        # 3. Return Subset
        if roles:
            roles_arr = x_arr[:,:,4:]
            x_arr = x_arr[:,:,:4]
            return TensorDataset(torch.from_numpy(x_arr), torch.from_numpy(c_arr), torch.from_numpy(roles_arr))
        
        return TensorDataset(torch.from_numpy(x_arr), torch.from_numpy(c_arr))

class ClimbsFeatureScaler:
    def __init__(self, weights_path: str | None = None):
        self.set_weights = False
        self.cond_features_scaler = MinMaxScaler(feature_range=(-1,1))
        self.hold_features_scaler = MinMaxScaler(feature_range=(-1,1))
        if weights_path and os.path.exists(weights_path):
            self.load_weights(weights_path)
    def save_weights(self, path: str):
        """Save weights to weights path."""
        state = {
            'cond_scaler': self.cond_features_scaler,
            'hold_scaler': self.hold_features_scaler
        }
        joblib.dump(state, path)
    
    def load_weights(self, path: str):
        """Load saved MinMaxScaler weights from the weights path."""
        state = joblib.load(path)
        self.cond_features_scaler = state['cond_scaler']
        self.hold_features_scaler = state['hold_scaler']
        self.set_weights = True
        
    def transform(self, climbs_to_fit: pd.DataFrame, holds_to_fit: pd.DataFrame):
        """Function for fitting the MinMax scalers to the climbs and holds dataframes and returning the transformed climbs and holds df"""
        # Fit preprocessing steps for climbs DF
        scaled_climbs = climbs_to_fit.copy()
        scaled_climbs = self._apply_log_transforms(scaled_climbs)
        # For holds DF
        scaled_holds = holds_to_fit.copy()
        scaled_holds = self._apply_hold_transforms(scaled_holds)
        
        if self.set_weights:
            scaled_climbs[['grade','quality','ascents','angle']] = self.cond_features_scaler.transform(scaled_climbs[['grade','quality','ascents','angle']])
            scaled_holds[['x','y','pull_x','pull_y']] = self.hold_features_scaler.transform(scaled_holds[['x','y','pull_x','pull_y']])
        else:
            scaled_climbs[['grade','quality','ascents','angle']] = self.cond_features_scaler.fit_transform(scaled_climbs[['grade','quality','ascents','angle']])
            scaled_holds[['x','y','pull_x','pull_y']] = self.hold_features_scaler.fit_transform(scaled_holds[['x','y','pull_x','pull_y']])
            self.set_weights = True
        
        return (scaled_climbs, scaled_holds)
    
    def _apply_log_transforms(self, dfc: pd.DataFrame) -> pd.DataFrame:
        """Covers Log transformation logic"""
        dfc['quality'] -= 3
        dfc['quality'] = np.log(1-dfc['quality'])
        dfc['ascents'] = np.log(dfc['ascents'])

        return dfc
    
    def _apply_hold_transforms(self, dfh: pd.DataFrame) -> pd.DataFrame:
        """Covers useability and is_foot embedding logic"""
        dfh['mult'] = dfh['useability'] / ((3 * dfh['is_foot'])+1)
        dfh['pull_x'] *= dfh['mult']
        dfh['pull_y'] *= dfh['mult']
        return dfh
    
    def transform_climb_features(self, climbs_to_transform: pd.DataFrame, to_df: bool = False):
        """Turn a series of conditional climb features into normalized features for the DDPM."""
        dfc = climbs_to_transform.copy()
        dfc = self._apply_log_transforms(dfc)
        if to_df:
            dfc[['grade','quality','ascents','angle']] = self.cond_features_scaler.transform(dfc[['grade','quality','ascents','angle']])
        else:
            dfc = self.cond_features_scaler.transform(dfc[['grade','quality','ascents','angle']])
            dfc = dfc.T
        return dfc
    
    def transform_hold_features(self, holds_to_transform: pd.DataFrame, to_df:bool=False):
        """Turn a series of hold features into normalized features for the DDPM."""
        dfh = holds_to_transform.copy()
        dfh = self._apply_hold_transforms(dfh)
        if to_df:
            dfh[['x','y','pull_x','pull_y']] = self.hold_features_scaler.transform(dfh[['x','y','pull_x','pull_y']])
        else:
            dfh = self.hold_features_scaler.transform(dfh[['x','y','pull_x','pull_y']])
            dfh = dfh.T

        return dfh

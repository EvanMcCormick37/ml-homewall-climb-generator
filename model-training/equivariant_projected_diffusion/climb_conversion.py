import numpy as np
import pandas as pd
import sqlite3
import json
import math
import torch
from dataclasses import dataclass
from torch_geometric.data import Data, InMemoryDataset
from sklearn.preprocessing import MinMaxScaler
import os
import joblib

@DeprecationWarning
class ClimbsFeatureArrayV1:
    def __init__(
            self,
            db_path: str = "data/storage.db",
            to_length: int = 20
        ):
        """Gets climbs from the climbs database and converts them into featured Sequence data for our DDPM."""
        self.db_path = db_path
        self.to_length = to_length
        self.scaler = MinMaxScaler(feature_range=(-1,1))

        with sqlite3.connect(db_path) as conn:
            query = "SELECT * FROM climbs WHERE ascents > 1"
            fdf = pd.read_sql_query(query, conn, index_col='id')

            all_holds = pd.read_sql_query("SELECT hold_index, x, y, pull_x, pull_y, useability, is_foot, wall_id FROM holds",conn)

            all_holds['mult'] = all_holds['useability'] / ((3 * all_holds['is_foot'])+1)
            all_holds['pull_x'] *= all_holds['mult']
            all_holds['pull_y'] *= all_holds['mult']
            all_holds['x'] /= 6
            all_holds['y'] /= 6

            self.holds_lookup = {
                wall_id: group.drop(columns=['wall_id','useability', 'is_foot', 'mult']).to_dict('index')
                for wall_id, group in all_holds.groupby('wall_id') 
            }

            fdf['holds'] = fdf['holds'].apply(json.loads)
            fdf = fdf[fdf['holds'].map(len) <= to_length].copy()
            fdf[['grade','quality','ascents']] = self.scaler.fit_transform(fdf[['grade','quality','ascents']])
            self.climbs_df = fdf

    def apply_wall_angle(self, angle, val):
        rads = math.radians(angle)
        print(val, rads)
        return val * math.cos(rads), val * math.sin(rads)

    def _zero_center_of_mass(self, arr, dim=2, start_holds=None):
        """Return the center-of-mass of the climbing hold positions"""
        if start_holds is not None:
            com = np.mean(arr[start_holds,:dim], axis=0)
        else:
            com = np.mean(arr[:,:dim], axis=0)
        arr[:,:dim] -= com
        
        return arr

    def _2d_features(self, h_data):
        return [h_data['x'],h_data['y'], h_data['pull_x'], h_data['pull_y']]
    
    def _3d_features(self, h_data, angle):
        y, z = self.apply_wall_angle(h_data['y'], angle)
        py, pz = self.apply_wall_angle(h_data['pull_y'], angle)
        return [h_data['x'], y, z, h_data['pull_x'], py, pz]
    
    def get_features(self, dim=2, continuous_only = True, augment_reflections=True):
        """Extract features from climbs"""
        assert dim in [2,3]

        x_out, cond_out = [], []

        for _, row in self.climbs_df.iterrows():
            climb_holds = row['holds']
            wall_id = row['wall_id']
            angle = row['angle']
            start_holds=[]
            try:
                wall_holds = self.holds_lookup[wall_id]
                features = []
                for i, (h_idx, role) in enumerate(climb_holds):
                    if role == 0:
                        start_holds.append(i)
                    h_data = wall_holds[h_idx]
                    feat = []
                    if dim==3:
                        feat = self._3d_features(h_data, angle)
                    elif dim==2:
                        feat = self._2d_features(h_data)
                    if not continuous_only:
                        role_vec = [0.0]*5
                        role_vec[role]=1.0
                        feat.extend(role_vec)
                    
                    if len(feat) > 0:
                        features.append(feat)
                
                f_arr = np.array(features)
                f_arr = self._zero_center_of_mass(f_arr, start_holds=start_holds)
                f_arr = np.array(sorted(f_arr, key=(lambda x: x[0]+x[1])))

                f_arr = self._zero_center_of_mass(f_arr)

                pad_length = self.to_length - len(f_arr)
                if pad_length > 0:
                    null_holds = np.tile(np.array([-2,-2,-2,-2]),(pad_length,1))
                    f_arr = np.concatenate([f_arr,null_holds], axis=0, dtype=np.float32)

                x_out.append(f_arr)
                cond_out.append([row['grade'], row['quality'], row['ascents'], angle/90])
            except KeyError:
                continue
        
        x_out = np.array(x_out)
        cond_out = np.array(cond_out)
        if augment_reflections:
            refl = x_out.copy()
            refl[:,:,[0, dim]] *= -1
            x_out = np.concatenate([x_out,refl],axis=0)
            cond_out = np.tile(cond_out,(2,1))
        
        return torch.FloatTensor(x_out), torch.FloatTensor(cond_out)

class ClimbsFeatureArray:
    def __init__(
            self,
            db_path: str = "data/storage.db",
            to_length: int = 20,
            scaler_weights_path: str | None = None,
        ):
        """Gets climbs from the climbs database and converts them into featured Sequence data for our DDPM."""
        self.db_path = db_path
        self.to_length = to_length
        self.scaler = ClimbsFeatureScaler(weights_path=scaler_weights_path)
        self.null_token = [-2,0,-2,0]

        with sqlite3.connect(db_path) as conn:
            climbs_to_fit = pd.read_sql_query("SELECT * FROM climbs WHERE ascents > 1", conn, index_col='id')
            climbs_to_fit['holds'] = climbs_to_fit['holds'].apply(json.loads)
            climbs_to_fit = climbs_to_fit[climbs_to_fit['holds'].map(len) <= to_length]

            holds_to_fit = pd.read_sql_query("SELECT hold_index, x, y, pull_x, pull_y, useability, is_foot, wall_id FROM holds", conn)
            
            scaled_climbs, scaled_holds = self.scaler.fit_transform(climbs_to_fit, holds_to_fit)
            self.hold_features_df = scaled_holds
            self.holds_lookup = {
                wall_id: group.drop(columns=['wall_id','useability', 'is_foot', 'mult']).to_dict('index')
                for wall_id, group in scaled_holds.groupby('wall_id')
            }
            self.climbs_df = scaled_climbs
        
        if scaler_weights_path:
            self.scaler.save_weights(scaler_weights_path)

    def apply_wall_angle(self, angle, val):
        rads = math.radians(angle)
        print(val, rads)
        return val * math.cos(rads), val * math.sin(rads)

    def _zero_center_of_mass(self, arr, dim=2, start_holds=None):
        """Return the center-of-mass of the climbing hold positions"""
        if start_holds is not None:
            com = np.mean(arr[start_holds,:dim], axis=0)
        else:
            com = np.mean(arr[:,:dim], axis=0)
        arr[:,:dim] -= com
        
        return arr

    def _2d_features(self, h_data):
        return [h_data['x'],h_data['y'], h_data['pull_x'], h_data['pull_y']]
    
    def _3d_features(self, h_data, angle):
        y, z = self.apply_wall_angle(h_data['y'], angle)
        py, pz = self.apply_wall_angle(h_data['pull_y'], angle)
        return [h_data['x'], y, z, h_data['pull_x'], py, pz]
    
    def get_features(self, dim=2, continuous_only = True, augment_reflections=True):
        """Extract features from climbs"""
        assert dim in [2,3]

        x_out, cond_out = [], []

        for _, row in self.climbs_df.iterrows():
            climb_holds = row['holds']
            wall_id = row['wall_id']
            angle = row['angle']
            start_holds=[]
            try:
                wall_holds = self.holds_lookup[wall_id]
                features = []
                for i, (h_idx, role) in enumerate(climb_holds):
                    if role == 0:
                        start_holds.append(i)
                    h_data = wall_holds[h_idx]
                    feat = []
                    if dim==3:
                        feat = self._3d_features(h_data, angle)
                    elif dim==2:
                        feat = self._2d_features(h_data)
                    if not continuous_only:
                        role_vec = [0.0]*5
                        role_vec[role]=1.0
                        feat.extend(role_vec)
                    
                    if len(feat) > 0:
                        features.append(feat)
                
                f_arr = np.array(features)
                f_arr = self._zero_center_of_mass(f_arr, start_holds=start_holds)
                f_arr = np.array(sorted(f_arr, key=(lambda x: x[0]+x[1])))

                f_arr = self._zero_center_of_mass(f_arr)

                pad_length = self.to_length - len(f_arr)
                if pad_length > 0:
                    null_holds = np.tile(np.array(self.null_token),(pad_length,1))
                    f_arr = np.concatenate([f_arr,null_holds], axis=0, dtype=np.float32)

                x_out.append(f_arr)
                cond_out.append([row['grade'], row['quality'], row['ascents'], angle/90])
            except KeyError:
                continue
        
        x_out = np.array(x_out)
        cond_out = np.array(cond_out)
        if augment_reflections:
            refl = x_out.copy()
            refl[:,:,[0, dim]] *= -1
            x_out = np.concatenate([x_out,refl],axis=0)
            cond_out = np.tile(cond_out,(2,1))
        
        return torch.FloatTensor(x_out), torch.FloatTensor(cond_out)

class ClimbsFeatureScaler:
    def __init__(self, weights_path: str | None = None):
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
        
    def fit_transform(self, climbs_to_fit: pd.DataFrame, holds_to_fit: pd.DataFrame):
        """Function for fitting the MinMax scalers to the climbs and holds dataframes and returning the transformed climbs and holds df"""
        # Fit preprocessing steps for climbs DF
        scaled_climbs = climbs_to_fit.copy()
        scaled_climbs = self._apply_log_transforms(scaled_climbs)
        scaled_climbs[['grade','quality','ascents','angle']] = self.cond_features_scaler.fit_transform(scaled_climbs[['grade','quality','ascents','angle']])
        # For holds DF
        scaled_holds = holds_to_fit.copy()
        scaled_holds = self._apply_hold_transforms(scaled_holds)
        scaled_holds[['x','y','pull_x','pull_y']] = self.hold_features_scaler.fit_transform(scaled_holds[['x','y','pull_x','pull_y']])
        
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

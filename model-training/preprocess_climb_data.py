import json
import numpy as np
from typing import List, Dict, Tuple

def encode_hold_features(hold_data: Dict) -> np.ndarray:
    """
    Convert a single hold's data into a 6-dimensional feature vector.
    
    Features: [norm_x, norm_y, pull_x, pull_y, useability, type]
    - type: 'hold' (hand-hold) -> 1, 'foot' (foot-only) -> 0
    """
    type_encoding = (1 if hold_data['type'] == 'hold' else 0)
    
    features = np.array([
        hold_data['norm_x'],
        hold_data['norm_y'],
        hold_data['pull_x'],
        hold_data['pull_y'],
        hold_data['useability'] / 10.0,  # Normalize useability to [0, 1]
        type_encoding
    ], dtype=np.float32)
    
    return features

def encode_null_hold() -> np.ndarray:
    """
    Encode the "no hold" case - limb not on wall.
    
    Uses [-1, -1, -1, -1, -1, -1] to place it far from any valid hold.
    """
    return np.array([-1, -1, 0, 0, -1, -1], dtype=np.float32)

def encode_position(holds_by_limb: List) -> np.ndarray:
    """
    Convert a single position (4 limbs) into a feature vector.
    
    Args:
        holds_by_limb: List of 4 elements [LH, RH, LF, RF], each either a hold dict or -1
    
    Returns:
        Feature vector of shape (4 * 6,) = (24,) - flattened representation
    """
    position_features = []
    
    for limb_hold in holds_by_limb:
        if limb_hold == -1:
            position_features.append(encode_null_hold())
        else:
            position_features.append(encode_hold_features(limb_hold))
    
    # Flatten: [LH_features, RH_features, LF_features, RF_features]
    return np.concatenate(position_features)

def encode_climb_sequence(climb_data: Dict) -> Tuple[np.ndarray, str, int]:
    """
    Convert an entire climb into a sequence of position embeddings.
    
    Args:
        climb_data: Single climb dict with 'sequence', 'name', 'grade'
    
    Returns:
        - positions: np.ndarray of shape (num_positions + 1, 24)
                    Last position is the STOP token (all null holds)
        - name: Climb name
        - grade: Climb grade as integer
    """
    sequence = climb_data['sequence']
    positions = []
    
    # Encode each position in the sequence
    for position in sequence:
        pos_embedding = encode_position(position['holdsByLimb'])
        positions.append(pos_embedding)
    
    # Append STOP token: position where all limbs are -1
    stop_token = encode_position([-1, -1, -1, -1])
    positions.append(stop_token)
    
    # Stack into 2D array: (num_positions + 1, 24)
    positions_array = np.stack(positions, axis=0)
    
    return positions_array, climb_data['name'], int(climb_data['grade'])

def load_and_preprocess_climbs(json_path: str) -> Tuple[List[np.ndarray], List[str], List[int]]:
    """
    Load all climbs from JSON and convert to feature tensors.
    
    Returns:
        - sequences: List of np.ndarrays, each of shape (num_positions + 1, 24)
        - names: List of climb names
        - grades: List of climb grades
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    sequences = []
    names = []
    grades = []
    
    for climb in data['climbs']:
        seq, name, grade = encode_climb_sequence(climb)
        sequences.append(seq)
        names.append(name)
        grades.append(grade)
    
    return sequences, names, grades

def mirror_climb(sequence: np.ndarray) -> np.ndarray:
    """
    Data augmentation: Mirror a climb left-to-right.
    
    Operations:
    - Swap LH <-> RH (indices 0-5 <-> 6-11)
    - Swap LF <-> RF (indices 12-17 <-> 18-23)
    - Negate norm_x (index 0, 6, 12, 18) and pull_x (index 2, 8, 14, 20)
    
    Args:
        sequence: np.ndarray of shape (num_positions, 24)
    
    Returns:
        Mirrored sequence of same shape
    """
    mirrored = sequence.copy()
    
    # Swap LH <-> RH
    lh = mirrored[:, 0:6].copy()
    rh = mirrored[:, 6:12].copy()
    mirrored[:, 0:6] = rh
    mirrored[:, 6:12] = lh
    
    # Swap LF <-> RF
    lf = mirrored[:, 12:18].copy()
    rf = mirrored[:, 18:24].copy()
    mirrored[:, 12:18] = rf
    mirrored[:, 18:24] = lf
    
    # Negate norm_x and pull_x for all limbs
    # LH: norm_x at 0, pull_x at 2
    # RH: norm_x at 6, pull_x at 8
    # LF: norm_x at 12, pull_x at 14
    # RF: norm_x at 18, pull_x at 20
    for limb_offset in [0, 6, 12, 18]:
        mirrored[:, limb_offset] = -mirrored[:, limb_offset]  # norm_x
        mirrored[:, limb_offset + 2] = -mirrored[:, limb_offset + 2]  # pull_x
    
    return mirrored

def translate_climb(sequence: np.ndarray) -> List[np.ndarray]:
    """
    Data Augmentation by translating a climb horizontally, while keeping it within [0,1].
    
    :param sequence: The climb sequence to be augmented
    :type sequence: np.ndarray
    :param direction: The direction to shift the climb
    :type direction: np.ndarray
    :return: The augmented climb.
    :rtype: ndarray[_AnyShape, dtype[Any]]
    """
    translated_left = sequence.copy()
    translated_right = sequence.copy()

    max_left = 1
    max_right = 1

    for limb_offset in [0,6,12,18]:
        limb_is_used = sequence[:,limb_offset]!=-1
        if np.any(limb_is_used):
            max_left = min(max_left,np.min(sequence[limb_is_used,limb_offset]))
            max_right = min(max_right,np.min(1-sequence[limb_is_used,limb_offset]))
    
    for limb_offset in [0,6,12,18]:
        limb_is_used = sequence[:,limb_offset]!=-1
        if np.any(limb_is_used):
            translated_left[limb_is_used,limb_offset] -= max_left
            translated_right[limb_is_used,limb_offset] += max_right

    return [translated_left, translated_right]

def augment_dataset(sequences: List[np.ndarray], names: List[str], grades: List[int]) -> Tuple[List[np.ndarray], List[str], List[int]]:
    """
    Apply data augmentation to all climbs by mirroring, translating max-left, and translating max-right.
    
    Returns augmented dataset with 4x the original size.
    """
    aug_sequences = sequences.copy()
    aug_names = names.copy()
    aug_grades = grades.copy()
    
    for seq, name, grade in zip(sequences, names, grades):
        print(name)
        mirrored = mirror_climb(seq)
        left, right = translate_climb(seq)
        m_right, m_left = translate_climb(mirrored)
        # I just want to check that I've created a valid symmetric group and nothing fishy is going on.
        assert mirror_climb(left) == m_left
        aug_sequences.extend([seq, left, right, mirrored, m_right, m_left])
        aug_names.extend([f"{name}",f"{name} (left)",f"{name} (right)", f"{name} (mirrored)", f"{name} (right=>mirrored)", f"{name} (left=>mirrored)"])
        aug_grades.extend([[grade]*6])
    
    return aug_sequences, aug_names, aug_grades

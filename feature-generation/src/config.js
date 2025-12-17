// Constants for Limb visual properties
export const LIMB_CONFIG = [
  { label: 'LH', color: '#ff6b6b', full: 'Left Hand', adjust: [-25,25]},  // 0: Left Hand (Blue)
  { label: 'RH', color: '#4ecdc4', full: 'Right Hand', adjust: [25,25]}, // 1: Right Hand (Red)
  // { label: 'LF', color: '#370064ff', full: 'Left Foot', adjust: [-25,-25]},  // 2: Left Foot (Blue)
  // { label: 'RF', color: '#00065cff', full: 'Right Foot', adjust: [25,-25]}  // 3: Right Foot (Red)
];

// Move cursor modes for the move annotation workflow
export const MOVE_CURSOR_MODES = {
  LH_START: 'lh_start',
  RH_START: 'rh_start', 
  LH_FINISH: 'lh_finish',
  RH_FINISH: 'rh_finish'
};

// Configuration for move cursor modes (similar to LIMB_CONFIG)
export const MOVE_CURSOR_CONFIG = {
  [MOVE_CURSOR_MODES.LH_START]: {
    label: 'LHS',
    color: '#ff6b6b',
    description: 'Left Hand Start'
  },
  [MOVE_CURSOR_MODES.RH_START]: {
    label: 'RHS', 
    color: '#4ecdc4',
    description: 'Right Hand Start'
  },
  [MOVE_CURSOR_MODES.LH_FINISH]: {
    label: 'LHF',
    color: '#c40000ff',
    description: 'Left Hand Finish (multi)'
  },
  [MOVE_CURSOR_MODES.RH_FINISH]: {
    label: 'RHF',
    color: '#0004ffff', 
    description: 'Right Hand Finish (multi)'
  }
};
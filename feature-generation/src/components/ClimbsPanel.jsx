import PropTypes from 'prop-types';
import { LIMB_CONFIG } from './config';
import { useState } from 'react';

export default function ClimbsPanel({
  useClimbParams
}) {
  const {
    position,
    currentClimb,
    climbName,
    climbGrade,
    climbs,
    setPosition,
    resetPosition,
    setCurrentClimb,
    setClimbs,
    setClimbName,
    setClimbGrade,
    addPositionToCurrentClimb,
    removeLastPositionFromCurrentClimb,
    addCurrentClimbToClimbs
  } = useClimbParams;
  const { holdsByLimb, activeLimb } = position;

  // Helper to switch active limb manually
  const handleCycleLimb = () => {
    setPosition(prev => ({
      ...prev,
      activeLimb: (prev.activeLimb + 1) % 4
    }));
  };

  // Helper to clear the entire sequence
  const handleResetClimb = () => {
    if (window.confirm("Are you sure you want to clear the current climb sequence?")) {
      setCurrentClimb([]);
      resetPosition();
    }
  };

  const handleDeleteAllClimbs = () => {
    if (window.confirm("WARNING. This will delete all climbs you have created. MAKE SURE YOU EXPORT BEFORE DELETION")) {
      if (window.confirm("SERIOUSLY. ARE YOU SURE YOU WANT TO DO THIS?")) {
        setClimbs([]);
        setCurrentClimb([]);
        setClimbName('');
        setClimbGrade('');
        resetPosition();
      }
    }
  }

  const handleRemoveLastPositionFromClimb = () => {
    setPosition(currentClimb[currentClimb.length-1] ?? {holdsByLimb: [-1,-1,-1,-1], activeLimb: 0 });
    removeLastPositionFromCurrentClimb();
  }

  const handleResetPosition = () => {
    resetPosition();
  };

  const handleSaveClimb = () => {
    addCurrentClimbToClimbs();
  };

  return (
    <div className="climbs-panel">
      <h3>Climb Builder</h3>

      {/* --- Current Position Section --- */}
      <div className="climbs-section">
        <h4>Current Position</h4>
        <div className="limb-grid">
          {LIMB_CONFIG.map((limb, index) => {
            const isActive = activeLimb === index;
            const hold = holdsByLimb[index];
            const hasHold = hold !== -1 && hold !== null;

            return (
              <div 
                key={limb.label} 
                className={`limb-card ${isActive ? 'active' : ''}`}
                style={{ borderColor: isActive ? limb.color : '#444' }}
                onClick={() => setPosition(prev => {
                  const newHoldsByLimb = [...prev.holdsByLimb];
                  newHoldsByLimb[index] = -1;
                  return { holdsByLimb: newHoldsByLimb, activeLimb: index }
                })}
              >
                <span style={{ color: limb.color, fontWeight: 'bold' }}>{limb.label}</span>
                <div className="hold-info">
                  {hasHold ? `Hold #${hold.hold_id}` : 'Empty'}
                </div>
              </div>
            );
          })}
        </div>
        
        <div className="button-group">
          <button className="btn" onClick={handleCycleLimb}>
            Next Limb (V)
          </button>
          <button className="btn btn-secondary" onClick={handleResetPosition}>
            Clear (C)
          </button>
        </div>
        
        <button className="btn btn-primary full-width" onClick={addPositionToCurrentClimb}>
          Add Position (Space)
        </button>
        <button className="btn btn-primary full-width" onClick={handleRemoveLastPositionFromClimb}>
          Rollback Last Position (R)
        </button>
      </div>

      <hr className="climbs-divider" />

      {/* --- Sequence Section --- */}
      <div className="climbs-section">
        <h4>Current Sequence</h4>
        <div className="climbs-stats">
          <div>Steps: <strong>{currentClimb.length}</strong></div>
        </div>

        {/* New Input Fields */}
        <div className="input-stack">
          <input 
            type="text" 
            className="climb-input" 
            placeholder="Climb Name"
            value={climbName}
            onChange={(e) => setClimbName(e.target.value)}
          />
          <input 
            type="text" 
            className="climb-input short" 
            placeholder="Grade"
            value={climbGrade}
            onChange={(e) => setClimbGrade(e.target.value)}
          />
        </div>
        
        <div className="button-group">
          <button 
            className="btn btn-save" 
            onClick={handleSaveClimb}
            disabled={currentClimb.length === 0}
          >
            Save Climb
          </button>
          <button 
            className="btn btn-danger" 
            onClick={handleResetClimb}
            disabled={currentClimb.length === 0}
          >
            Discard
          </button>
        </div>
        <button 
            className="btn btn-danger" 
            onClick={handleDeleteAllClimbs}
            disabled={climbs.length === 0}
          >
            Delete All Climbs
          </button>
      </div>

      <hr className="climbs-divider" />

      <div className="climbs-footer">
        <span>Total Saved Climbs: {climbs.length}</span>
      </div>
    </div>
  );
}

ClimbsPanel.propTypes = {
  climbs: PropTypes.array.isRequired,
  currentClimb: PropTypes.array.isRequired,
  setCurrentClimb: PropTypes.func.isRequired,
  position: PropTypes.shape({
    holdsByLimb: PropTypes.array.isRequired,
    activeLimb: PropTypes.number.isRequired
  }).isRequired,
  setPosition: PropTypes.func.isRequired,
  addPositionToCurrentClimb: PropTypes.func.isRequired,
  addCurrentClimbToClimbs: PropTypes.func.isRequired
};
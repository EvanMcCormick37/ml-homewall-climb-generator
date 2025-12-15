import PropTypes from 'prop-types';
import { LIMB_CONFIG } from './config';

export default function ClimbsPanel({
  useClimbParams
}) {
  const {  
    climbs,             // From useClimbs
    currentClimb,       // From useClimbs
    setCurrentClimb,    // From useClimbs
    position,           // From useClimbs
    setPosition,        // From useClimbs
    resetPosition,
    addPositionToCurrentClimb, // From useClimbs
    addCurrentClimbToClimbs    // From useClimbs
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

  return (
    <div className="climbs-panel" style={styles.panel}>
      <h3 style={styles.header}>Climb Builder</h3>

      {/* --- Current Position Section --- */}
      <div style={styles.section}>
        <h4 style={styles.subHeader}>Current Position</h4>
        <div style={styles.limbGrid}>
          {LIMB_CONFIG.map((limb, index) => {
            const isActive = activeLimb === index;
            const hold = holdsByLimb[index];
            const hasHold = hold !== -1 && hold !== null;

            return (
              <div 
                key={limb.label} 
                style={{
                  ...styles.limbCard,
                  borderColor: isActive ? limb.color : '#444',
                  backgroundColor: isActive ? 'rgba(255,255,255,0.05)' : 'transparent',
                  opacity: isActive ? 1 : 0.7
                }}
                onClick={() => setPosition(prev => ({ ...prev, activeLimb: index }))} // Click to select limb
              >
                <span style={{ color: limb.color, fontWeight: 'bold' }}>{limb.label}</span>
                <div style={styles.holdInfo}>
                  {hasHold ? `Hold #${hold.hold_id}` : 'Empty'}
                </div>
              </div>
            );
          })}
        </div>
        
        <div style={styles.buttonGroup}>
          <button style={styles.button} onClick={handleCycleLimb}>
            Next Limb (V)
          </button>
          <button style={styles.buttonSecondary} onClick={handleResetPosition}>
            Clear Position (C)
          </button>
        </div>
        
        <button style={styles.actionButton} onClick={addPositionToCurrentClimb}>
          Add Position to Sequence (Space)
        </button>
      </div>

      <hr style={styles.divider} />

      {/* --- Sequence Section --- */}
      <div style={styles.section}>
        <h4 style={styles.subHeader}>Current Sequence</h4>
        <div style={styles.stats}>
          <div>Steps: <strong>{currentClimb.length}</strong></div>
        </div>
        
        <div style={styles.buttonGroup}>
          <button 
            style={{...styles.button, backgroundColor: '#009165', color: 'white'}} 
            onClick={addCurrentClimbToClimbs}
            disabled={currentClimb.length === 0}
          >
            Save Climb (Enter)
          </button>
          <button 
            style={styles.buttonDanger} 
            onClick={handleResetClimb}
            disabled={currentClimb.length === 0}
          >
            Discard
          </button>
        </div>
      </div>

      <hr style={styles.divider} />

      {/* --- Total Stats --- */}
      <div style={styles.footer}>
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

// Simple inline styles for rapid implementation
const styles = {
  panel: {
    backgroundColor: '#1e1e2f',
    color: '#eee',
    padding: '15px',
    borderRadius: '8px',
    width: '300px',
    fontFamily: 'sans-serif',
    boxShadow: '0 4px 6px rgba(0,0,0,0.3)',
  },
  header: {
    margin: '0 0 15px 0',
    textAlign: 'center',
    borderBottom: '1px solid #444',
    paddingBottom: '10px'
  },
  subHeader: {
    marginTop: 0,
    marginBottom: '10px',
    fontSize: '0.9rem',
    textTransform: 'uppercase',
    color: '#aaa'
  },
  section: {
    marginBottom: '15px'
  },
  limbGrid: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: '8px',
    marginBottom: '10px'
  },
  limbCard: {
    border: '2px solid',
    borderRadius: '6px',
    padding: '8px',
    textAlign: 'center',
    cursor: 'pointer',
    transition: 'all 0.2s'
  },
  holdInfo: {
    fontSize: '0.8rem',
    marginTop: '4px'
  },
  buttonGroup: {
    display: 'flex',
    gap: '8px',
    marginBottom: '8px'
  },
  button: {
    flex: 1,
    padding: '8px',
    cursor: 'pointer',
    backgroundColor: '#444',
    color: 'white',
    border: 'none',
    borderRadius: '4px'
  },
  buttonSecondary: {
    flex: 1,
    padding: '8px',
    cursor: 'pointer',
    backgroundColor: 'transparent',
    color: '#aaa',
    border: '1px solid #666',
    borderRadius: '4px'
  },
  buttonDanger: {
    flex: 1,
    padding: '8px',
    cursor: 'pointer',
    backgroundColor: '#632b2b',
    color: '#ffcccc',
    border: 'none',
    borderRadius: '4px'
  },
  actionButton: {
    width: '100%',
    padding: '10px',
    cursor: 'pointer',
    backgroundColor: '#3399ff',
    color: 'white',
    fontWeight: 'bold',
    border: 'none',
    borderRadius: '4px',
    marginTop: '5px'
  },
  divider: {
    border: '0',
    borderTop: '1px solid #444',
    margin: '15px 0'
  },
  stats: {
    marginBottom: '10px',
    fontSize: '0.9rem'
  },
  footer: {
    fontSize: '0.8rem',
    color: '#888',
    textAlign: 'center'
  }
};
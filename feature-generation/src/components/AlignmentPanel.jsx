import PropTypes from 'prop-types';

function AlignmentPanel({ holdParams }) {
  const {
    alignment,
    setAlignment,
    onReset,
    onApply 
  } = holdParams;
  
  const { scale, offsetX, offsetY } = alignment;
  
  const handleScaleChange = (value) => {
    setAlignment(prev => ({ ...prev, scale: parseFloat(value) }));
  };
  
  const handleOffsetXChange = (value) => {
    setAlignment(prev => ({ ...prev, offsetX: parseInt(value, 10) }));
  };
  
  const handleOffsetYChange = (value) => {
    setAlignment(prev => ({ ...prev, offsetY: parseInt(value, 10) }));
  };
  
  return (
    <div className="alignment-panel">
      <h4>🎯 Hold Alignment</h4>
      <p className="alignment-hint">
        Adjust to align JSON holds with image
      </p>
      
      <div className="alignment-row">
        <label>Scale:</label>
        <input
          type="range"
          min="0.1"
          max="3"
          step="0.01"
          value={scale}
          onChange={(e) => handleScaleChange(e.target.value)}
        />
        <input
          type="number"
          min="0.1"
          max="3"
          step="0.01"
          value={scale}
          onChange={(e) => handleScaleChange(e.target.value)}
        />
      </div>
      
      <div className="alignment-row">
        <label>Offset X:</label>
        <input
          type="range"
          min="-500"
          max="500"
          value={offsetX}
          onChange={(e) => handleOffsetXChange(e.target.value)}
        />
        <input
          type="number"
          value={offsetX}
          onChange={(e) => handleOffsetXChange(e.target.value)}
        />
      </div>
      
      <div className="alignment-row">
        <label>Offset Y:</label>
        <input
          type="range"
          min="-500"
          max="500"
          value={offsetY}
          onChange={(e) => handleOffsetYChange(e.target.value)}
        />
        <input
          type="number"
          value={offsetY}
          onChange={(e) => handleOffsetYChange(e.target.value)}
        />
      </div>
      
      <div className="alignment-buttons">
        <button className="action-btn" onClick={onReset}>
          Reset
        </button>
        <button className="action-btn apply" onClick={onApply}>
          Apply to Holds
        </button>
      </div>
    </div>
  );
}

AlignmentPanel.propTypes = {
  alignment: PropTypes.shape({
    scale: PropTypes.number.isRequired,
    offsetX: PropTypes.number.isRequired,
    offsetY: PropTypes.number.isRequired
  }).isRequired,
  setAlignment: PropTypes.func.isRequired,
  onReset: PropTypes.func.isRequired,
  onApply: PropTypes.func.isRequired
};

export default AlignmentPanel;

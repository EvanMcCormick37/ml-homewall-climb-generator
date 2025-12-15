import PropTypes from 'prop-types';

function Toolbar({
  mode,
  setMode,
  zoom, 
  imageInputRef,
  jsonInputRef,
  onZoomIn,
  onZoomOut,
  onZoomChange,
  onFit,
  onImageLoad,
  onJsonLoad,
  onExport,
  onClear,
  status,
}) {
  
  const modes = [
    { id: 'view', label: '👁️ View', key: '1' },
    { id: 'hold', label: '➕ Hold', key: '2' },
    { id: 'foot', label: '➕ Foot', key: '3' },
    { id: 'remove', label: '➖ Remove', key: '4' },
    { id: 'pan', label: '✋ Pan', key: '5' }
  ];
  
  const handleImageChange = (e) => {
    const file = e.target.files?.[0];
    if (file) onImageLoad(file);
  };
  
  const handleJsonChange = (e) => {
    const file = e.target.files?.[0];
    if (file) onJsonLoad(file);
  };
  
  return (
    <header className="toolbar-header">
      <h1>🧗 Hold Annotator</h1>
      
      <div className="toolbar">
        {/* File inputs */}
        <div className="toolbar-group">
          <input
            ref={imageInputRef}
            type="file"
            accept="image/*"
            onChange={handleImageChange}
            hidden
          />
          <button
            className="action-btn"
            onClick={() => imageInputRef.current?.click()}
          >
            📷 Image
          </button>
          
          <input
            ref={jsonInputRef}
            type="file"
            accept=".json"
            onChange={handleJsonChange}
            hidden
          />
          <button
            className="action-btn"
            onClick={() => jsonInputRef.current?.click()}
          >
            📁 JSON
          </button>
        </div>
        
        {/* Mode buttons */}
        <div className="toolbar-group">
          {modes.map(({ id, label }) => (
            <button
              key={id}
              className={`mode-btn ${mode === id ? 'active' : ''}`}
              onClick={() => setMode(id)}
            >
              {label}
            </button>
          ))}
        </div>
        
        {/* Zoom controls */}
        <div className="toolbar-group">
          <span className="toolbar-label">Zoom:</span>
          <button className="action-btn small" onClick={onZoomOut}>−</button>
          <input
            type="range"
            min="10"
            max="200"
            value={zoom * 100}
            onChange={(e) => onZoomChange(e.target.value / 100)}
            className="zoom-slider"
          />
          <button className="action-btn small" onClick={onZoomIn}>+</button>
          <button className="action-btn" onClick={onFit}>Fit</button>
        </div>
        
        {/* Export controls */}
        <div className="toolbar-group">
          <button className="action-btn" onClick={onExport}>
            💾 Export
          </button>
          <button className="action-btn danger" onClick={onClear}>
            🗑️ Clear
          </button>
        </div>
        
        {/* Status */}
        <span className="status">{status}</span>
      </div>
    </header>
  );
}

Toolbar.propTypes = {
  mode: PropTypes.string.isRequired,
  setMode: PropTypes.func.isRequired,
  zoom: PropTypes.number.isRequired,
  onZoomIn: PropTypes.func.isRequired,
  onZoomOut: PropTypes.func.isRequired,
  onZoomChange: PropTypes.func.isRequired,
  onFit: PropTypes.func.isRequired,
  onImageLoad: PropTypes.func.isRequired,
  onJsonLoad: PropTypes.func.isRequired,
  onExport: PropTypes.func.isRequired,
  onClear: PropTypes.func.isRequired,
  status: PropTypes.string.isRequired
};

export default Toolbar;

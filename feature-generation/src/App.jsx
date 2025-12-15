import { useState, useRef, useCallback, useEffect } from 'react';
import { Toolbar, CanvasArea, AlignmentPanel, HelpPanel } from './components';
import { useHolds, useViewTransform } from './hooks/useHoldAnnotator';
import './App.css';

function App() {
  // Image state
  const [image, setImage] = useState(null);
  const [imageDimensions, setImageDimensions] = useState({ width: 0, height: 0 });
  
  // UI mode
  const [mode, setMode] = useState('view'); // 'view', 'add', 'remove', 'pan'
  
  // Custom hooks for holds and view transform
  const {
    holds,
    alignment,
    setAlignment,
    loadHolds,
    addHold,
    removeHold,
    findHoldAt,
    applyAlignment,
    resetAlignment,
    clearHolds,
    exportHolds
  } = useHolds(imageDimensions);
  
  const {
    viewTransform,
    setViewTransform,
    zoom,
    setZoom,
    fitToContainer
  } = useViewTransform();
  
  // Refs
  const wrapperRef = useRef(null);
  
  // Load image file
  const handleImageLoad = useCallback((file) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        setImage(img);
        setImageDimensions({ width: img.width, height: img.height });
        
        // Auto-fit to window after image loads
        if (wrapperRef.current) {
          const rect = wrapperRef.current.getBoundingClientRect();
          fitToContainer(rect, img.width, img.height);
        }
      };
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
  }, [fitToContainer]);
  
  // Load JSON file
  const handleJsonLoad = useCallback((file) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const data = JSON.parse(e.target.result);
        loadHolds(data);
      } catch (err) {
        alert('Error loading JSON: ' + err.message);
      }
    };
    reader.readAsText(file);
  }, [loadHolds]);
  
  // Fit to window
  const fitToWindow = useCallback(() => {
    if (!image || !wrapperRef.current) return;
    const rect = wrapperRef.current.getBoundingClientRect();
    fitToContainer(rect, imageDimensions.width, imageDimensions.height);
  }, [image, imageDimensions, fitToContainer]);
  
  // Handle zoom with optional center point
  const handleZoom = useCallback((delta, centerX = null, centerY = null) => {
    zoom(delta, centerX, centerY);
  }, [zoom]);
  
  // Export JSON and download
  const handleExport = useCallback(() => {
    const data = exportHolds();
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'holds_annotated.json';
    a.click();
    URL.revokeObjectURL(url);
  }, [exportHolds]);
  
  // Clear with confirmation
  const handleClear = useCallback(() => {
    if (window.confirm('Remove all holds?')) {
      clearHolds();
    }
  }, [clearHolds]);
  
  // Apply alignment with notification
  const handleApplyAlignment = useCallback(() => {
    const count = applyAlignment();
    alert(`Applied alignment to ${count} holds.`);
  }, [applyAlignment]);
  
  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.target.tagName === 'INPUT') return;
      
      switch (e.key) {
        case '1': setMode('view'); break;
        case '2': setMode('add'); break;
        case '3': setMode('remove'); break;
        case '4': setMode('pan'); break;
        case 'f':
        case 'F': fitToWindow(); break;
        default: break;
      }
    };
    
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [fitToWindow]);
  
  // Status text
  const statusText = `${mode.charAt(0).toUpperCase() + mode.slice(1)} | Holds: ${holds.length} | Zoom: ${Math.round(viewTransform.zoom * 100)}%`;
  
  return (
    <div className="app">
      <Toolbar
        mode={mode}
        setMode={setMode}
        zoom={viewTransform.zoom}
        onZoomIn={() => handleZoom(1.2)}
        onZoomOut={() => handleZoom(0.9)}
        onZoomChange={setZoom}
        onFit={fitToWindow}
        onImageLoad={handleImageLoad}
        onJsonLoad={handleJsonLoad}
        onExport={handleExport}
        onClear={handleClear}
        status={statusText}
      />
      
      <CanvasArea
        ref={wrapperRef}
        image={image}
        imageDimensions={imageDimensions}
        holds={holds}
        mode={mode}
        viewTransform={viewTransform}
        setViewTransform={setViewTransform}
        alignment={alignment}
        onAddHold={addHold}
        onRemoveHold={removeHold}
        onFindHold={findHoldAt}
        onZoom={handleZoom}
      />
      
      {image && holds.length > 0 && (
        <AlignmentPanel
          alignment={alignment}
          setAlignment={setAlignment}
          onReset={resetAlignment}
          onApply={handleApplyAlignment}
        />
      )}
      
      <HelpPanel />
    </div>
  );
}

export default App;

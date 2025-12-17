import { useState, useRef, useCallback, useEffect } from 'react';
import { Toolbar, CanvasArea, AlignmentPanel, HelpPanel, ClimbsPanel, MovesPanel } from './components';
import { useClimbs, useHolds, useMoves, useViewTransform } from './hooks/useHoldAnnotator';
import './App.css';

function App() {
  // Image state
  const [image, setImage] = useState(null);
  const [imageDimensions, setImageDimensions] = useState({ width: 0, height: 0 });

  // Input refs for JSON and Image Upload.
  const imageInputRef = useRef(null);
  const jsonInputRef = useRef(null);
  
  // UI mode
  const [mode, setMode] = useState('view'); // 'view', 'hold', 'foot', 'climb', 'remove', 'pan'
  
  // Custom hooks for holds and view transform
  const holdParams = useHolds(imageDimensions);
  const climbParams = useClimbs();
  const moveParams = useMoves();
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
        holdParams.loadHolds(data);
      } catch (err) {
        alert('Error loading JSON: ' + err.message);
      }
    };
    reader.readAsText(file);
  }, [holdParams.loadHolds]);
  
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
  const handleExport = useCallback((data) => {
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${data.metadata?.wall_name??'wall'}-${data.metadata?.data_type??'stuff'}-${data.metadata?.exported??'time'}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, []);

  const handleExportHolds = useCallback(()=>handleExport(holdParams.exportHolds()),[holdParams.exportHolds]);
  const handleExportClimbs = useCallback(()=>handleExport(climbParams.exportClimbs()),[climbParams.exportClimbs]);
  const handleExportMoves = useCallback(()=>handleExport(moveParams.exportMoves()),[moveParams.exportMoves]);
  const handleExportAll = useCallback(()=>{
    const holds_data = holdParams.exportHolds();
    const climbs_data = climbParams.exportClimbs();
    const moves_data = moveParams.exportMoves();
    const metadata = {
      ...holds_data.metadata,
      data_type: 'all',
      num_sequences: climbs_data?.metadata?.num_climbs ?? 0,
      num_moves: (climbs_data?.metadata?.num_moves ?? 0) + (moves_data?.metadata?.num_moves ?? 0),
    }
    const all_data = {
      metadata,
      holds: holds_data.holds,
      sequences: climbs_data.climbs,
      movesets: moves_data.moves
    }
    handleExport(all_data);
  }, [holdParams.exportHolds, climbParams.exportClimbs, moveParams.exportMoves])
  const exportFunctions = [
    handleExportHolds,
    handleExportClimbs,
    handleExportMoves,
    handleExportAll
  ];
  
  // Clear with confirmation
  const handleClear = useCallback(() => {
    if (window.confirm('Remove all holds?')) {
      holdParams.clearHolds();
    }
    if (jsonInputRef.current.value){
      jsonInputRef.current.value = null;
    }
  }, [holdParams.clearHolds]);
  
  // Apply alignment with notification
  const handleApplyAlignment = useCallback(() => {
    const count = holdParams.applyAlignment();
    alert(`Applied alignment to ${count} holds.`);
  }, [holdParams.applyAlignment]);
  
  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.target.tagName === 'INPUT') return;
      
      switch (e.key) {
        case '1': setMode('view'); break;
        case '2': setMode('hold'); break;
        case '3': setMode('foot'); break;
        case '4': setMode('climb'); break;
        case '5': setMode('move'); break;
        case '6': setMode('remove'); break;
        case '7': setMode('pan'); break;
        case 'w':
        case 'W': fitToWindow(); break;
        default: break;
      }
    };
    
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [fitToWindow]);
  
  // Status text
  const statusText = `${mode.charAt(0).toUpperCase() + mode.slice(1)} | Holds: ${holdParams.holds.length} | Zoom: ${Math.round(viewTransform.zoom * 100)}%`;
  
  return (
    <div className="app">
      <Toolbar
        mode={mode}
        setMode={setMode}
        zoom={viewTransform.zoom}
        imageInputRef={imageInputRef}
        jsonInputRef={jsonInputRef}
        onZoomIn={() => handleZoom(1.2)}
        onZoomOut={() => handleZoom(0.9)}
        onZoomChange={setZoom}
        onFit={fitToWindow}
        onImageLoad={handleImageLoad}
        onJsonLoad={handleJsonLoad}
        onClear={handleClear}
        status={statusText}
        exportFunctions = {exportFunctions}
      />
      
      <CanvasArea
        ref={wrapperRef}
        image={image}
        imageDimensions={imageDimensions}
        mode={mode}
        viewTransform={viewTransform}
        setViewTransform={setViewTransform}
        onZoom={handleZoom}
        climbParams={climbParams}
        moveParams={moveParams}
        holdParams={holdParams}
      />
      
      {image && holdParams.holds.length > 0 && (
        <AlignmentPanel
          holdParams={holdParams}
        />
      )}
      {mode==='climb' && (<ClimbsPanel climbParams={climbParams}/>)}
      {mode==='move' && (<MovesPanel
                          moveParams = {moveParams}
                          holds = {holdParams.holds}
                        />)
      }
      <HelpPanel mode={mode}/>
    </div>
  );
}

export default App;

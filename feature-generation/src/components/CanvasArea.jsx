import { useRef, useEffect, useCallback, forwardRef, useState } from 'react';
import PropTypes from 'prop-types';
import { MOVE_CURSOR_MODES, MOVE_CURSOR_CONFIG, LIMB_CONFIG } from '../config';

const CanvasArea = forwardRef(function CanvasArea({
  image,
  imageDimensions,
  mode,
  viewTransform,
  setViewTransform,
  onZoom,
  climbParams,
  moveParams,
  holdParams,
}, wrapperRef) {

  const canvasRef = useRef(null);

  // Size of Climb Hold direction indicators. TODO: Add slider to toggle arrow size.
  const [ arrowSize, setArrowSize ] = useState(2);

  // Pan drag state
  const panDragRef = useRef({
    isDragging: false,
    startX: 0,
    startY: 0,
    startViewX: 0,
    startViewY: 0
  });
  
  // Add hold drag state
  const [addHoldState, setAddHoldState] = useState({
    isDragging: false,
    holdX: 0,
    holdY: 0,
    dragX: 0,
    dragY: 0
  });

  const {
    holds,
    alignment,
    addHold,
    removeHold,
    findHoldAt,
  } = holdParams;

  // Climb State
  const {
    position, 
    currentClimb,
    holdsUsedInCurrentClimb,
    setPosition,
    resetPosition,
    addPositionToCurrentClimb, 
    removeLastPositionFromCurrentClimb,
    addCurrentClimbToClimbs,
  } = climbParams;

  // Move State (NEW)
  const {
    currentMove,
    moveCursorMode,
    setMoveCursorMode,
    handleMoveHoldClick,
    addCurrentMoveToMoves,
    resetCurrentMove,
  } = moveParams;
  
  // Get image coordinates from mouse event
  const getImageCoords = useCallback((e) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    
    const rect = canvas.getBoundingClientRect();
    const scaleX = imageDimensions.width / rect.width;
    const scaleY = imageDimensions.height / rect.height;
    
    return {
      x: Math.round((e.clientX - rect.left) * scaleX),
      y: Math.round((e.clientY - rect.top) * scaleY)
    };
  }, [imageDimensions]);
  
  // Calculate pull direction and useability from drag vector
  const calculateHoldParams = useCallback((holdX, holdY, dragX, dragY) => {
    const dx = holdX - dragX;
    const dy = holdY - dragY;
    const magnitude = Math.sqrt(dx * dx + dy * dy);
    
    const pull_x = dx / magnitude;
    const pull_y = dy / magnitude;
    const useability = Math.min(10, Math.round(magnitude / 25));
    
    return { pull_x, pull_y, useability };
  }, []);
  
  const getUseabilityColor = (useability) => {
    const t = (useability)/10; 
    let r,g,b;
    if (t < 0.5) { 
      const t2 = t * 2;
      r = 255;
      g = Math.round(60 + 160*t2);
      b = 60;
    } else {
      const t2 = (1-t)*2;
      r = Math.round(60 + 195*t2);
      g = 220;
      b = 60;
    }
    return `rgb(${r}, ${g}, ${b})`;
  }

  // Draw canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const { width, height } = imageDimensions;
    
    canvas.width = width || 800;
    canvas.height = height || 600;
    
    // Clear
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    if (!image) {
      ctx.fillStyle = '#666';
      ctx.font = '24px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('Load an image to begin', canvas.width / 2, canvas.height / 2);
      return;
    }
    
    ctx.drawImage(image, 0, 0);
    
    // Draw holds with alignment transform
    const { scale, offsetX, offsetY } = alignment;
    
    holds.forEach((hold) => {
      const holdInClimb = (mode==='climb' && holdsUsedInCurrentClimb && holdsUsedInCurrentClimb.length > 0 && holdsUsedInCurrentClimb.includes(hold.hold_id))
      
      const x = hold.pixel_x * scale + offsetX;
      const y = hold.pixel_y * scale + offsetY;
      const radius = holdInClimb? 45 * scale : 15 * scale; 

      // Useful when we add feet back in
      // Draw hold circle
      if((mode !== 'climb') || holdInClimb){
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, 2 * Math.PI);
        ctx.strokeStyle = hold.type === 'hold' ? (holdInClimb? '#ff0000ff' : '#009165ff') : '#63008aff';
        ctx.lineWidth = holdInClimb? 8 : 3;
        ctx.stroke();
      }
      
      // Draw pull direction arrow
      if ((mode !== 'climb') || holdInClimb && hold.pull_x !== undefined && hold.pull_y !== undefined) {
        const arrowLength = 10 + 30 * arrowSize * scale * (hold.useability / 10); 
        const endX = x + hold.pull_x * arrowLength;
        const endY = y + hold.pull_y * arrowLength;

        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(endX, endY);
        ctx.strokeStyle = getUseabilityColor(hold.useability);
        ctx.lineWidth = arrowSize;
        ctx.stroke();

        const headLength = arrowLength/5.0;
        const angle = Math.atan2(hold.pull_y, hold.pull_x);
        ctx.beginPath();
        ctx.moveTo(
          endX + (arrowSize/2.0) * Math.cos(angle - Math.PI / 4),
          endY + (arrowSize/2.0) * Math.sin(angle - Math.PI / 4)
        );
        ctx.lineTo(
          endX - headLength * Math.cos(angle - Math.PI / 4),
          endY - headLength * Math.sin(angle - Math.PI / 4)
        );
        ctx.moveTo(
          endX + (arrowSize/2.0) * Math.cos(angle + Math.PI / 4),
          endY + (arrowSize/2.0) * Math.sin(angle + Math.PI / 4)
        );
        ctx.lineTo(
          endX - headLength * Math.cos(angle + Math.PI / 4),
          endY - headLength * Math.sin(angle + Math.PI / 4)
        );
        ctx.stroke();
      }

      // --- Highlight holds selected in current Position State (CLIMB MODE) ---
      if (mode === 'climb') {
        position.holdsByLimb.forEach((limbHold, idx)=>{
          if (limbHold >= 0 && limbHold === hold.hold_id) {
          const limbInfo = LIMB_CONFIG[idx];
          const [x_adj, y_adj] = [x+limbInfo.adjust[0],y+limbInfo.adjust[1]]
          
          // Draw selection ring
          ctx.beginPath();
          ctx.arc(x_adj, y_adj, radius + 8, 0, 2 * Math.PI);
          ctx.strokeStyle = limbInfo.color;
          ctx.lineWidth = 4;
          ctx.stroke();

          // Draw Limb Label (LH, RH, etc)
          ctx.fillStyle = limbInfo.color;
          ctx.font = 'bold 16px sans-serif';
          ctx.textAlign = 'right';
          ctx.fillText(limbInfo.label, x_adj, y_adj);
          ctx.moveTo(x,y)
        }
        })
      }

      // --- Highlight holds selected in current Move State (MOVE MODE) ---
      if (mode === 'move' && currentMove) {
        let isSelected = false;
        let label = '';
        let color = '';
        let isFinish = false;

        // Check LH Start
        if (currentMove.lh_start === hold.hold_id) {
          isSelected = true;
          label = 'LHS';
          color = MOVE_CURSOR_CONFIG[MOVE_CURSOR_MODES.LH_START].color;
        }
        // Check RH Start
        if (currentMove.rh_start === hold.hold_id) {
          isSelected = true;
          label = label ? label + '/RHS' : 'RHS';
          color = color || MOVE_CURSOR_CONFIG[MOVE_CURSOR_MODES.RH_START].color;
        }
        // Check LH Finish
        if (currentMove.lh_finish.includes(hold.hold_id)) {
          isSelected = true;
          isFinish = true;
          label = label ? label + '/LHF' : 'LHF';
          color = color || MOVE_CURSOR_CONFIG[MOVE_CURSOR_MODES.LH_FINISH].color;
        }
        // Check RH Finish
        if (currentMove.rh_finish.includes(hold.hold_id)) {
          isSelected = true;
          isFinish = true;
          label = label ? label + '/RHF' : 'RHF';
          color = color || MOVE_CURSOR_CONFIG[MOVE_CURSOR_MODES.RH_FINISH].color;
        }

        if (isSelected) {
          // Draw selection ring (dashed for finish holds)
          ctx.beginPath();
          ctx.arc(x, y, radius + 12, 0, 2 * Math.PI);
          ctx.strokeStyle = color;
          ctx.lineWidth = 6;
          if (isFinish) {
            ctx.setLineDash([5, 3]);
          }
          ctx.stroke();
          ctx.setLineDash([]);

          // Draw label above hold
          ctx.fillStyle = color;
          ctx.font = 'bold 14px sans-serif';
          ctx.textAlign = 'center';
          ctx.fillText(label, x, y - radius - 15);
        }
      }
      
      // Standard Hold ID Label
      ctx.fillStyle = 'rgba(0, 0, 0, 0.85)';
      const text = hold.hold_id.toString();
      ctx.font = 'bold 14px sans-serif';
      const textWidth = ctx.measureText(text).width;
      ctx.fillRect(x - textWidth / 2 - 4, y - 9, textWidth + 8, 18);
      
      ctx.fillStyle = hold.type === 'hold' ? 'white' : 'purple';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(text, x, y);
    });
    
    // Draw add-hold preview if dragging
    if (addHoldState.isDragging) {
      const { holdX, holdY, dragX, dragY } = addHoldState;
      const { pull_x, pull_y, useability } = calculateHoldParams(holdX, holdY, dragX, dragY);
      const dragColor = getUseabilityColor(useability);
      
      ctx.beginPath();
      ctx.arc(holdX, holdY, 15, 0, 2 * Math.PI);
      ctx.strokeStyle = dragColor;
      ctx.lineWidth = 3;
      ctx.setLineDash([5, 5]);
      ctx.stroke();
      ctx.setLineDash([]);
      
      ctx.beginPath();
      ctx.moveTo(dragX, dragY);
      ctx.lineTo(holdX, holdY);
      ctx.strokeStyle = dragColor;
      ctx.lineWidth = 2;
      ctx.stroke();
      
      const angle = Math.atan2(holdY - dragY, holdX - dragX);
      const headLength = 12;
      ctx.beginPath();
      ctx.moveTo(holdX, holdY);
      ctx.lineTo(
        holdX - headLength * Math.cos(angle - Math.PI / 6),
        holdY - headLength * Math.sin(angle - Math.PI / 6)
      );
      ctx.moveTo(holdX, holdY);
      ctx.lineTo(
        holdX - headLength * Math.cos(angle + Math.PI / 6),
        holdY - headLength * Math.sin(angle + Math.PI / 6)
      );
      ctx.stroke();
      
      ctx.font = 'bold 16px sans-serif';
      ctx.fillStyle = dragColor;
      ctx.textAlign = 'left';
      ctx.fillText(`Useability: ${useability}`, dragX + 10, dragY - 10);
      
      const pullAngle = Math.atan2(-pull_y, pull_x) * 180 / Math.PI; 
      ctx.font = '12px sans-serif';
      ctx.fillText(`Pull: ${pullAngle.toFixed(0)}°`, dragX + 10, dragY + 10);
    }

    // Draw current cursor mode indicator for move mode
    if (mode === 'move' && moveCursorMode) {
      const config = MOVE_CURSOR_CONFIG[moveCursorMode];
      ctx.fillStyle = config.color;
      ctx.font = 'bold 16px sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText(`Mode: ${config.label}`, 10, 25);
    }
    
  }, [
    image,
    imageDimensions,
    holds,
    alignment,
    addHoldState,
    calculateHoldParams,
    mode,
    position,
    currentClimb,
    currentMove,
    moveCursorMode
  ]);
  
  // Handle mouse down
  const handleMouseDown = useCallback((e) => {
    if (!image) return;
    
    const { x, y } = getImageCoords(e);
    
    if (mode === 'pan') {
      panDragRef.current = {
        isDragging: true,
        startX: e.clientX,
        startY: e.clientY,
        startViewX: viewTransform.x,
        startViewY: viewTransform.y
      };
    } else if (mode === 'hold' || mode === 'foot') {
      setAddHoldState({
        isDragging: true,
        holdX: x,
        holdY: y,
        dragX: x,
        dragY: y,
      });
    }
  }, [image, mode, getImageCoords, viewTransform, position]);
  
  // Handle mouse move
  const handleMouseMove = useCallback((e) => {

    // Pan mode dragging
    if (panDragRef.current.isDragging) {
      const drag = panDragRef.current;
      setViewTransform(prev => ({
        ...prev,
        x: drag.startViewX + (e.clientX - drag.startX),
        y: drag.startViewY + (e.clientY - drag.startY)
      }));
      return;
    }
    
    // Add mode dragging
    if (addHoldState.isDragging) {
      const { x, y } = getImageCoords(e);
      setAddHoldState(prev => ({
        ...prev,
        dragX: x,
        dragY: y
      }));
    }
  }, [addHoldState.isDragging, getImageCoords, setViewTransform, mode]);

  // Handle mouse up
  const handleMouseUp = useCallback((e) => {
    if (panDragRef.current.isDragging) {
      panDragRef.current.isDragging = false;
      return;
    }
    
    if (addHoldState.isDragging) {
      const { holdX, holdY, dragX, dragY } = addHoldState;
      const { pull_x, pull_y, useability } = calculateHoldParams(holdX, holdY, dragX, dragY);
      
      addHold(holdX, holdY, pull_x, pull_y, useability, mode);
      
      setAddHoldState({
        isDragging: false,
        holdX: 0,
        holdY: 0,
        dragX: 0,
        dragY: 0
      });
    }
  }, [addHoldState, calculateHoldParams, addHold, mode]);
  
  // Handle click
  const handleClick = useCallback((e) => {
    if (!image || mode === 'pan' || mode === 'hold' || mode === 'foot') return;
    
    const { x, y } = getImageCoords(e);
    
    if (mode === 'remove') {
      removeHold(x, y);
    } else if (mode === 'view') {
      const hold = findHoldAt(x, y);
      if (hold) {
        const pullAngle = hold.pull_x !== undefined 
          ? `${(Math.atan2(-hold.pull_y, hold.pull_x) * 180 / Math.PI).toFixed(1)}°`
          : 'N/A';
        alert(
          `Hold #${hold.hold_id}\n` +
          `Pixel: (${hold.pixel_x}, ${hold.pixel_y})\n` +
          `Normalized: (${hold.norm_x.toFixed(3)}, ${hold.norm_y.toFixed(3)})\n` +
          `Pull Direction: ${pullAngle}\n` +
          `Useability: ${hold.useability ?? 'N/A'}`
        );
      }
    } else if (mode === 'climb') {
      const foundHold = findHoldAt(x,y);
      if (foundHold) {
        setPosition((prev) => {
          const newHolds = [...prev.holdsByLimb];
          newHolds[prev.activeLimb] = foundHold.hold_id;
          return { holdsByLimb: newHolds, activeLimb: (prev.activeLimb + 1) % 2 }
        });
      }
    } else if (mode === 'move') {
      // NEW: Handle move mode clicks
      const foundHold = findHoldAt(x, y);
      if (foundHold && handleMoveHoldClick) {
        handleMoveHoldClick(foundHold.hold_id);
      }
    }
  }, [image, mode, getImageCoords, removeHold, findHoldAt, setPosition, handleMoveHoldClick]);

  const handleKeyDown = useCallback((e) => {
    if (mode === 'climb') {
      if (e.target.tagName === 'INPUT'){
        return;
      }
      e.preventDefault();
      switch (e.key) {
        case 'a':
        case 'A':
          setPosition((prev)=>{
            const newHoldsByLimb = [...prev.holdsByLimb];
            newHoldsByLimb[0] = -1;
            return {holdsByLimb: newHoldsByLimb, activeLimb: 0 }});
          break;
        case 's':
        case 'S':
          setPosition((prev)=>{
            const newHoldsByLimb = [...prev.holdsByLimb];
            newHoldsByLimb[1] = -1;
            return {holdsByLimb: newHoldsByLimb, activeLimb: 1 }});
          break;
        case 'x':
        case 'X':
          setPosition((prev)=>{
            const newHoldsByLimb = [...prev.holdsByLimb];
            const previouslyActiveLimb = (prev.activeLimb + 1) % 2;
            newHoldsByLimb[previouslyActiveLimb] = -1;
            return { holdsByLimb: newHoldsByLimb, activeLimb: previouslyActiveLimb }
          });
          break;
        case 'c':
        case 'C':
          resetPosition();
          break;
        case 'v':
        case 'V': 
          setPosition((prev) => ({
            ...prev,
            activeLimb: (prev.activeLimb + 1) % 2
          }));
          break;
        case 'r':
        case 'R':
          setPosition({holdsByLimb: currentClimb[currentClimb.length - 1] ?? [-1,-1], activeLimb: 0});
          removeLastPositionFromCurrentClimb();
          break;
        case ' ':
          addPositionToCurrentClimb()
          break;
      }
    } else if (mode === 'move') {
      // NEW: Handle move mode keyboard shortcuts
      if (e.target.tagName === 'INPUT') {
        return;
      }
      
      switch (e.key) {
        case 'q':
        case 'Q':
          e.preventDefault();
          setMoveCursorMode(MOVE_CURSOR_MODES.LH_START);
          break;
        case 'w':
        case 'W':
          e.preventDefault();
          setMoveCursorMode(MOVE_CURSOR_MODES.RH_START);
          break;
        case 'e':
        case 'E':
          e.preventDefault();
          setMoveCursorMode(MOVE_CURSOR_MODES.LH_FINISH);
          break;
        case 'r':
        case 'R':
          e.preventDefault();
          setMoveCursorMode(MOVE_CURSOR_MODES.RH_FINISH);
          break;
        case ' ':
          e.preventDefault();
          addCurrentMoveToMoves();
          break;
        case 'c':
        case 'C':
          e.preventDefault();
          resetCurrentMove();
          break;
      }
    }
  },[mode, position, currentClimb, setPosition, resetPosition, addPositionToCurrentClimb, 
     removeLastPositionFromCurrentClimb, setMoveCursorMode, addCurrentMoveToMoves, resetCurrentMove]);
  
  // Global mouse events
  useEffect(() => {
    const handleGlobalMouseMove = (e) => handleMouseMove(e);
    const handleGlobalMouseUp = (e) => handleMouseUp(e);
    const handleGlobalKeyDown = (e) => handleKeyDown(e);
    
    window.addEventListener('mousemove', handleGlobalMouseMove);
    window.addEventListener('mouseup', handleGlobalMouseUp);
    window.addEventListener('keydown', handleGlobalKeyDown);
    
    return () => {
      window.removeEventListener('mousemove', handleGlobalMouseMove);
      window.removeEventListener('mouseup', handleGlobalMouseUp);
      window.removeEventListener('keydown', handleGlobalKeyDown);
    };
  }, [handleMouseMove, handleMouseUp, handleKeyDown]);
  
  // Wheel zoom
  const handleWheel = useCallback((e) => {
    e.preventDefault();
    const wrapper = wrapperRef.current;
    if (!wrapper) return;
    
    const rect = wrapper.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    onZoom(delta, mouseX, mouseY);
  }, [wrapperRef, onZoom]);
  
  useEffect(() => {
    const wrapper = wrapperRef.current;
    if (!wrapper) return;
    wrapper.addEventListener('wheel', handleWheel, { passive: false });
    return () => wrapper.removeEventListener('wheel', handleWheel);
  }, [wrapperRef, handleWheel]);
  
  const getCursor = () => {
    if (addHoldState.isDragging) return 'crosshair';
    if (panDragRef.current.isDragging) return 'grabbing';
    
    switch (mode) {
      case 'pan': return 'grab';
      case 'hold': return 'crosshair';
      case 'foot': return 'crosshair';
      case 'climb': return 'crosshair';
      case 'move': return 'crosshair';
      case 'remove': return 'pointer';
      default: return 'default';
    }
  };
  
  const { zoom, x, y } = viewTransform;
  const { width, height } = imageDimensions;
  
  return (
    <div className="canvas-wrapper" ref={wrapperRef}>
      <div
        className="canvas-container"
        style={{
          transform: `translate(${x}px, ${y}px)`
        }}
      >
        <canvas
          ref={canvasRef}
          style={{
            width: (width || 800) * zoom,
            height: (height || 600) * zoom,
            cursor: getCursor()
          }}
          onClick={handleClick}
          onMouseDown={handleMouseDown}
        />
      </div>
    </div>
  );
});

CanvasArea.propTypes = {
  image: PropTypes.instanceOf(Image),
  imageDimensions: PropTypes.shape({
    width: PropTypes.number.isRequired,
    height: PropTypes.number.isRequired
  }).isRequired,
  holds: PropTypes.array.isRequired,
  mode: PropTypes.string.isRequired,
  viewTransform: PropTypes.shape({
    zoom: PropTypes.number.isRequired,
    x: PropTypes.number.isRequired,
    y: PropTypes.number.isRequired
  }).isRequired,
  setViewTransform: PropTypes.func.isRequired,
  alignment: PropTypes.shape({
    scale: PropTypes.number.isRequired,
    offsetX: PropTypes.number.isRequired,
    offsetY: PropTypes.number.isRequired
  }).isRequired,
  addHold: PropTypes.func.isRequired,
  removeHold: PropTypes.func.isRequired,
  findHoldAt: PropTypes.func.isRequired,
  onSetNewClimb: PropTypes.func,
  onZoom: PropTypes.func.isRequired,
  useClimbParams: PropTypes.object.isRequired,
  useMovesParams: PropTypes.object // NEW: Optional move params
};

export default CanvasArea;
import { useState, useCallback, useMemo } from 'react';
import { MOVE_CURSOR_MODES } from '../config';

/**
 * Custom hook for managing hold annotation  and operations.
 * Encapsulates all hold-related  and CRUD operations.
 */
export function useHolds(imageDimensions) {
  const [holds, setHolds] = useState([]);
  const [nextId, setNextId] = useState(0);
  const [alignment, setAlignment] = useState({
    scale: 1,
    offsetX: 0,
    offsetY: 0
  });
  
  // Load holds from JSON data
  const loadHolds = useCallback((data) => {
    const loadedHolds = data.holds || [];
    setHolds(loadedHolds);
    setNextId(Math.max(...loadedHolds.map(h => h.hold_id), -1) + 1);
  }, []);
  
  // Add a new hold at position with pull direction and useability
  const addHold = useCallback((x, y, pull_x, pull_y, useability, footOnly) => {
    const { width, height } = imageDimensions;
    if (!width || !height) return;
    
    const newHold = {
      hold_id: nextId,
      pixel_x: x,
      pixel_y: y,
      norm_x: x / width,
      norm_y: 1 - (y / height),
      pull_x: pull_x,
      pull_y: pull_y,
      useability: useability,
      type: footOnly ? 'foot' : 'hold', // Track whether this is a hand-hold or foot-only hold.
      manual: true
    };
    
    setHolds(prev => [...prev, newHold]);
    setNextId(prev => prev + 1);
  }, [nextId, imageDimensions]);
  
  // Remove hold nearest to position
  const removeHold = useCallback((x, y, radius = 40) => {
    const { scale, offsetX, offsetY } = alignment;
    
    setHolds(prev => {
      let minDist = Infinity;
      let nearestIdx = -1;
      
      prev.forEach((hold, idx) => {
        const hx = hold.pixel_x * scale + offsetX;
        const hy = hold.pixel_y * scale + offsetY;
        const dist = Math.sqrt((hx - x) ** 2 + (hy - y) ** 2);
        
        if (dist < minDist && dist < radius) {
          minDist = dist;
          nearestIdx = idx;
        }
      });
      
      if (nearestIdx >= 0) {
        return prev.filter((_, idx) => idx !== nearestIdx);
      }
      return prev;
    });
  }, [alignment]);
  
  // Find hold at position
  const findHoldAt = useCallback((x, y, radius = 40) => {
    const { scale, offsetX, offsetY } = alignment;
    let nearest = null;
    let minDist = Infinity;
    
    for (const hold of holds) {
      const hx = hold.pixel_x * scale + offsetX;
      const hy = hold.pixel_y * scale + offsetY;
      const dist = Math.sqrt((hx - x) ** 2 + (hy - y) ** 2);
      
      if (dist < minDist && dist < radius) {
        minDist = dist;
        nearest = hold;
      }
    }
    return nearest;
  }, [holds, alignment]);
  
  // Apply alignment transformation permanently
  const applyAlignment = useCallback(() => {
    const { scale, offsetX, offsetY } = alignment;
    const { width, height } = imageDimensions;
    
    if (!width || !height) return;
    
    setHolds(prev => prev.map(hold => ({
      ...hold,
      pixel_x: Math.round(hold.pixel_x * scale + offsetX),
      pixel_y: Math.round(hold.pixel_y * scale + offsetY),
      norm_x: (hold.pixel_x * scale + offsetX) / width,
      norm_y: 1 - ((hold.pixel_y * scale + offsetY) / height)
    })));
    
    setAlignment({ scale: 1, offsetX: 0, offsetY: 0 });
    return holds.length;
  }, [alignment, imageDimensions, holds.length]);
  
  // Reset alignment to defaults
  const resetAlignment = useCallback(() => {
    setAlignment({ scale: 1, offsetX: 0, offsetY: 0 });
  }, []);
  
  // Clear all holds
  const clearHolds = useCallback(() => {
    setHolds([]);
    setNextId(0);
  }, []);
  
  // Export holds as JSON-ready object
  const exportHolds = useCallback(() => {
    // Sort top-to-bottom, left-to-right and renumber
    const sortedHolds = [...holds]
      .sort((a, b) => b.norm_y - a.norm_y || a.norm_x - b.norm_x)
      .map((hold, idx) => ({ ...hold, hold_id: idx }));
    
    // Update  with new IDs
    setHolds(sortedHolds);
    
    return {
      metadata: {
        wall_name: "Sideways Wall",
        data_type: "Holds",
        num_holds: sortedHolds.length,
        image_dimensions: [imageDimensions.width, imageDimensions.height],
        exported: new Date().toISOString()
      },
      holds: sortedHolds
    };
  }, [holds, imageDimensions]);
  
  return {
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
  };
}

/**
 * Custom hook for managing view transform (pan/zoom).
 */
export function useViewTransform() {
  const [viewTransform, setViewTransform] = useState({
    zoom: 1,
    x: 0,
    y: 0
  });
  
  const zoom = useCallback((delta, centerX = null, centerY = null) => {
    setViewTransform(prev => {
      const newZoom = Math.max(0.1, Math.min(3, prev.zoom * delta));
      
      if (centerX !== null && centerY !== null) {
        const beforeX = (centerX - prev.x) / prev.zoom;
        const beforeY = (centerY - prev.y) / prev.zoom;
        return {
          zoom: newZoom,
          x: centerX - beforeX * newZoom,
          y: centerY - beforeY * newZoom
        };
      }
      
      return { ...prev, zoom: newZoom };
    });
  }, []);
  
  const setZoom = useCallback((newZoom) => {
    setViewTransform(prev => ({ ...prev, zoom: newZoom }));
  }, []);
  
  const pan = useCallback((dx, dy) => {
    setViewTransform(prev => ({
      ...prev,
      x: prev.x + dx,
      y: prev.y + dy
    }));
  }, []);
  
  const fitToContainer = useCallback((containerRect, imageWidth, imageHeight) => {
    if (!imageWidth || !imageHeight) return;
    
    const scaleX = containerRect.width / imageWidth;
    const scaleY = containerRect.height / imageHeight;
    const newZoom = Math.min(scaleX, scaleY) * 0.95;
    
    setViewTransform({
      zoom: newZoom,
      x: (containerRect.width - imageWidth * newZoom) / 2,
      y: (containerRect.height - imageHeight * newZoom) / 2
    });
  }, []);
  
  return {
    viewTransform,
    setViewTransform,
    zoom,
    setZoom,
    pan,
    fitToContainer
  };
}

export function useClimbs() {
  const [climbs, setClimbs] = useState([]);
  const [climbName, setClimbName] = useState('');
  const [climbGrade, setClimbGrade] = useState('');
  const [currentClimb, setCurrentClimb] = useState([]);
  const [position, setPosition] = useState({
    holdsByLimb: [-1, -1], // [LeftHand, RightHand]. -1 or null means not using a hold.
    activeLimb: 0
  });

  // --- NEW LOGIC START ---
  const holdsUsedInCurrentClimb = useMemo(() => {
    const allHolds = currentClimb.flat();
    const validHolds = allHolds.filter(h => h !== -1 && h !== null);

    return Array.from(new Set(validHolds)).sort((a, b) => a - b);
  }, [currentClimb]);

  const resetPosition = useCallback(() => (setPosition({
    holdsByLimb: [-1, -1],
    activeLimb: 0
  })), []);

  const addPositionToCurrentClimb = useCallback(() => {
    setCurrentClimb((prev) => ([...prev, [...position.holdsByLimb]]));
    resetPosition();
  }, [position]);

  const removeLastPositionFromCurrentClimb = useCallback(() => {
    setCurrentClimb((prev) => (prev.slice(0, -1) ?? []));
  }, []);

  const addCurrentClimbToClimbs = useCallback(() => {
    setClimbs((prev) => ([...prev, {
      name: climbName,
      grade: climbGrade,
      sequence: currentClimb
    }]));
    setCurrentClimb([]);
    setClimbName('');
    setClimbGrade('');
    resetPosition();
  }, [climbName, climbGrade, currentClimb, resetPosition]);

  const exportClimbs = useCallback(() => {
    const formattedClimbs = climbs.map((climb) => ({
      name: climb.name,
      grade: climb.grade,
      sequence: climb.sequence,
      num_moves: (climb.sequence.length - 1)
    }));
    return {
      metadata: {
        wall_name: "Sideways Wall",
        data_type: "Climb",
        num_climbs: climbs.length,
        num_moves: formattedClimbs.reduce((sum, climb) => (sum + climb.num_moves), 0),
        exported: new Date().toISOString()
      },
      climbs: formattedClimbs
    };
  }, [climbs]);

  return {
    position,
    currentClimb,
    holdsUsedInCurrentClimb,
    climbs,
    climbName,
    climbGrade,
    setPosition,
    resetPosition,
    setCurrentClimb,
    setClimbName,
    setClimbGrade,
    setClimbs,
    addPositionToCurrentClimb,
    removeLastPositionFromCurrentClimb,
    addCurrentClimbToClimbs,
    exportClimbs
  };
}

export function useMoves() {
  // Current move being constructed
  const [currentMove, setCurrentMove] = useState({
    lh_start: null,      // Single hold_id or null
    rh_start: null,      // Single hold_id or null
    lh_finish: [],       // Array of hold_ids (no duplicates)
    rh_finish: [],       // Array of hold_ids (no duplicates)
  });
  
  // Active cursor mode for move annotation
  const [moveCursorMode, setMoveCursorMode] = useState(MOVE_CURSOR_MODES.LH_START);
  
  // All saved moves
  const [moves, setMoves] = useState([]);

  // Reset current move to empty state
  const resetCurrentMove = useCallback(() => {
    setCurrentMove({
      lh_start: null,
      rh_start: null,
      lh_finish: [],
      rh_finish: [],
    });
  }, []);

  // Set a start hold (LH or RH)
  const setStartHold = useCallback((side, holdId) => {
    setCurrentMove(prev => ({
      ...prev,
      [side === 'lh' ? 'lh_start' : 'rh_start']: holdId
    }));
  }, []);

  // Toggle a finish hold (add if not present, remove if present)
  const toggleFinishHold = useCallback((side, holdId) => {
    setCurrentMove(prev => {
      const key = side === 'lh' ? 'lh_finish' : 'rh_finish';
      const currentFinishes = prev[key];
      
      if (currentFinishes.includes(holdId)) {
        // Remove if already present
        return {
          ...prev,
          [key]: currentFinishes.filter(id => id !== holdId)
        };
      } else {
        // Add if not present
        return {
          ...prev,
          [key]: [...currentFinishes, holdId]
        };
      }
    });
  }, []);

  // Handle a hold click based on current cursor mode
  const handleMoveHoldClick = useCallback((holdId) => {
    switch (moveCursorMode) {
      case MOVE_CURSOR_MODES.LH_START:
        setStartHold('lh', holdId);
        break;
      case MOVE_CURSOR_MODES.RH_START:
        setStartHold('rh', holdId);
        break;
      case MOVE_CURSOR_MODES.LH_FINISH:
        toggleFinishHold('lh', holdId);
        break;
      case MOVE_CURSOR_MODES.RH_FINISH:
        toggleFinishHold('rh', holdId);
        break;
      default:
        break;
    }
  }, [moveCursorMode, setStartHold, toggleFinishHold]);

  // Check if current move is valid (has at least one start and one finish)
  const isCurrentMoveValid = useCallback(() => {
    const hasStart = currentMove.lh_start !== null || currentMove.rh_start !== null;
    const hasFinish = currentMove.lh_finish.length > 0 || currentMove.rh_finish.length > 0;
    return hasStart && hasFinish;
  }, [currentMove]);

  // Add current move to moves list
  const addCurrentMoveToMoves = useCallback(() => {
    if (!isCurrentMoveValid()) return false;
    
    setMoves(prev => [...prev, { ...currentMove }]);
    resetCurrentMove();
    return true;
  }, [currentMove, isCurrentMoveValid, resetCurrentMove]);

  // Remove a move from the list by index
  const removeMove = useCallback((index) => {
    setMoves(prev => prev.filter((_, i) => i !== index));
  }, []);

  // Clear all moves
  const clearMoves = useCallback(() => {
    setMoves([]);
    resetCurrentMove();
  }, [resetCurrentMove]);

  // Load moves from JSON data
  const loadMoves = useCallback((data) => {
    const loadedMoves = data.moves || [];
    setMoves(loadedMoves);
  }, []);

  // Export moves as JSON-ready object
  const exportMoves = useCallback(() => {
    const formattedMoves = moves.map((move, idx) => ({
      move_id: idx,
      lh_start: move.lh_start,
      rh_start: move.rh_start,
      lh_finish: move.lh_finish,
      rh_finish: move.rh_finish,
      num_finish_options: move.lh_finish.length + move.rh_finish.length
    }));

    return {
      metadata: {
        wall_name: "Sideways Wall",
        data_type: "Moves",
        num_moves: formattedMoves.length,
        total_finish_options: formattedMoves.reduce((sum, move) => sum + move.num_finish_options, 0),
        exported: new Date().toISOString()
      },
      moves: formattedMoves
    };
  }, [moves]);

  return {
    // Current move state
    currentMove,
    setCurrentMove,
    resetCurrentMove,
    
    // Cursor mode
    moveCursorMode,
    setMoveCursorMode,
    
    // Move actions
    setStartHold,
    toggleFinishHold,
    handleMoveHoldClick,
    isCurrentMoveValid,
    addCurrentMoveToMoves,
    
    // Moves collection
    moves,
    setMoves,
    removeMove,
    clearMoves,
    loadMoves,
    exportMoves
  };
}
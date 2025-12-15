import { useState, useCallback } from 'react';

/**
 * Custom hook for managing hold annotation state and operations.
 * Encapsulates all hold-related state and CRUD operations.
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
    
    // Update state with new IDs
    setHolds(sortedHolds);
    
    return {
      metadata: {
        wall_name: "Home Spray Wall",
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

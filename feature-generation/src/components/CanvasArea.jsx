import { useRef, useEffect, useCallback, forwardRef } from 'react';
import PropTypes from 'prop-types';

const CanvasArea = forwardRef(function CanvasArea({
  image,
  imageDimensions,
  holds,
  mode,
  viewTransform,
  setViewTransform,
  alignment,
  onAddHold,
  onRemoveHold,
  onFindHold,
  onZoom
}, wrapperRef) {
  const canvasRef = useRef(null);
  const dragStateRef = useRef({
    isDragging: false,
    startX: 0,
    startY: 0,
    startViewX: 0,
    startViewY: 0
  });
  
  // Draw canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const { width, height } = imageDimensions;
    
    // Set canvas size
    canvas.width = width || 800;
    canvas.height = height || 600;
    
    // Clear
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    if (!image) {
      // Placeholder text
      ctx.fillStyle = '#666';
      ctx.font = '24px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('Load an image to begin', canvas.width / 2, canvas.height / 2);
      return;
    }
    
    // Draw image
    ctx.drawImage(image, 0, 0);
    
    // Draw holds with alignment transform
    const { scale, offsetX, offsetY } = alignment;
    
    holds.forEach(hold => {
      const x = hold.pixel_x * scale + offsetX;
      const y = hold.pixel_y * scale + offsetY;
      const radius = Math.max(12, Math.sqrt((hold.area || 500) / Math.PI) * scale);
      
      // Circle
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, 2 * Math.PI);
      ctx.strokeStyle = hold.manual ? '#00aaff' : '#00ff88';
      ctx.lineWidth = 3;
      ctx.stroke();
      
      // Label background
      ctx.fillStyle = 'rgba(0, 0, 0, 0.85)';
      const text = hold.hold_id.toString();
      ctx.font = 'bold 14px sans-serif';
      const textWidth = ctx.measureText(text).width;
      ctx.fillRect(x - textWidth / 2 - 4, y - 9, textWidth + 8, 18);
      
      // Label text
      ctx.fillStyle = 'white';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(text, x, y);
    });
  }, [image, imageDimensions, holds, alignment]);
  
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
  
  // Get pixel color from canvas
  const getPixelColor = useCallback((x, y) => {
    const canvas = canvasRef.current;
    if (!canvas) return [128, 128, 128];
    
    try {
      const ctx = canvas.getContext('2d');
      const data = ctx.getImageData(x, y, 1, 1).data;
      return [data[0], data[1], data[2]];
    } catch {
      return [128, 128, 128];
    }
  }, []);
  
  // Handle canvas click
  const handleClick = useCallback((e) => {
    if (!image || mode === 'pan' || dragStateRef.current.isDragging) return;
    
    const { x, y } = getImageCoords(e);
    
    switch (mode) {
      case 'add':
        onAddHold(x, y, getPixelColor(x, y));
        break;
      case 'remove':
        onRemoveHold(x, y);
        break;
      case 'view': {
        const hold = onFindHold(x, y);
        if (hold) {
          alert(
            `Hold #${hold.hold_id}\n` +
            `Pixel: (${hold.pixel_x}, ${hold.pixel_y})\n` +
            `Normalized: (${hold.norm_x.toFixed(3)}, ${hold.norm_y.toFixed(3)})\n` +
            `Area: ${hold.area} px²`
          );
        }
        break;
      }
      default:
        break;
    }
  }, [image, mode, getImageCoords, getPixelColor, onAddHold, onRemoveHold, onFindHold]);
  
  // Handle mouse down (for pan)
  const handleMouseDown = useCallback((e) => {
    if (mode !== 'pan') return;
    
    dragStateRef.current = {
      isDragging: true,
      startX: e.clientX,
      startY: e.clientY,
      startViewX: viewTransform.x,
      startViewY: viewTransform.y
    };
  }, [mode, viewTransform]);
  
  // Handle mouse move (for pan)
  useEffect(() => {
    const handleMouseMove = (e) => {
      const drag = dragStateRef.current;
      if (!drag.isDragging) return;
      
      setViewTransform(prev => ({
        ...prev,
        x: drag.startViewX + (e.clientX - drag.startX),
        y: drag.startViewY + (e.clientY - drag.startY)
      }));
    };
    
    const handleMouseUp = () => {
      dragStateRef.current.isDragging = false;
    };
    
    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
    
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [setViewTransform]);
  
  // Handle wheel zoom
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
  
  // Attach wheel listener (needs passive: false)
  useEffect(() => {
    const wrapper = wrapperRef.current;
    if (!wrapper) return;
    
    wrapper.addEventListener('wheel', handleWheel, { passive: false });
    return () => wrapper.removeEventListener('wheel', handleWheel);
  }, [wrapperRef, handleWheel]);
  
  // Cursor style based on mode
  const getCursor = () => {
    switch (mode) {
      case 'pan': return dragStateRef.current.isDragging ? 'grabbing' : 'grab';
      case 'add': return 'crosshair';
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
  onAddHold: PropTypes.func.isRequired,
  onRemoveHold: PropTypes.func.isRequired,
  onFindHold: PropTypes.func.isRequired,
  onZoom: PropTypes.func.isRequired
};

export default CanvasArea;

# Hold Annotator - React App

A React-based tool for annotating climbing holds on wall images, with support for loading/exporting JSON data and aligning hold overlays to different image sizes.

## Project Structure

```
src/
├── App.jsx                 # Main application component
├── App.css                 # Global styles
├── components/
│   ├── index.js            # Component exports
│   ├── Toolbar.jsx         # Top toolbar with buttons and controls
│   ├── CanvasArea.jsx      # Canvas rendering and interaction handling
│   ├── AlignmentPanel.jsx  # Hold alignment controls (scale/offset)
│   └── HelpPanel.jsx       # Keyboard shortcuts help
└── hooks/
    ├── index.js            # Hook exports
    └── useHoldAnnotator.js # Custom hooks for hold & view state
```

## Setup

This project assumes a Vite + React setup. To use:

1. Create a new Vite project:
   ```bash
   npm create vite@latest hold-annotator -- --template react
   cd hold-annotator
   ```

2. Copy the `src/` contents into your project's `src/` folder

3. Install dependencies:
   ```bash
   npm install prop-types
   ```

4. Run the dev server:
   ```bash
   npm run dev
   ```

## Features

- **Load Images**: Support for any image format (PNG, JPG, etc.)
- **Load/Export JSON**: Compatible with the hold detection pipeline output
- **Pan & Zoom**: Mouse wheel zoom, drag-to-pan
- **Add/Remove Holds**: Click to add new holds or remove existing ones
- **Alignment Controls**: Scale and offset holds to match different image sizes
- **Keyboard Shortcuts**: 1-4 for modes, F to fit, scroll to zoom

## Component API

### `<Toolbar>`
Top toolbar with file inputs, mode buttons, zoom controls, and export actions.

### `<CanvasArea>`
Main canvas for rendering the wall image and hold overlays. Handles all mouse interactions.

### `<AlignmentPanel>`
Floating panel for adjusting hold scale and offset when JSON doesn't match image size.

### `<HelpPanel>`
Displays keyboard shortcuts.

## Custom Hooks

### `useHolds(imageDimensions)`
Manages hold state, CRUD operations, and alignment transformations.

Returns:
- `holds` - Array of hold objects
- `alignment` - Current {scale, offsetX, offsetY}
- `addHold(x, y, color)` - Add hold at position
- `removeHold(x, y)` - Remove nearest hold
- `findHoldAt(x, y)` - Get hold at position
- `applyAlignment()` - Bake alignment into coordinates
- `exportHolds()` - Get JSON-ready data

### `useViewTransform()`
Manages pan/zoom viewport state.

Returns:
- `viewTransform` - Current {zoom, x, y}
- `zoom(delta, cx, cy)` - Zoom by factor
- `fitToContainer(rect, w, h)` - Fit image to container

## JSON Format

```json
{
  "metadata": {
    "wall_name": "Home Spray Wall",
    "num_holds": 150,
    "image_dimensions": [2000, 1505]
  },
  "holds": [
    {
      "hold_id": 0,
      "pixel_x": 500,
      "pixel_y": 200,
      "norm_x": 0.25,
      "norm_y": 0.87,
      "area": 1500,
      "color_rgb": [120, 180, 90],
      "confidence": 0.8,
      "manual": false
    }
  ]
}
```

## Extending

To add new features:

1. **New hold properties**: Extend the hold object in `useHolds` and add UI in `AlignmentPanel`
2. **New tools**: Add mode to `Toolbar` and handle in `CanvasArea.handleClick`
3. **Different export formats**: Modify `exportHolds()` in `useHoldAnnotator.js`

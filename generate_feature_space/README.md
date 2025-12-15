# Climbing Hold Detection Pipeline

## Overview

This pipeline semi-automatically extracts (x, y) coordinates for climbing holds from a wall photo. It provides:

1. **Automated detection** using color segmentation
2. **Manual verification** via web-based annotation tool
3. **JSON output** ready for your ML climb generation model

## Files Included

| File | Description |
|------|-------------|
| `hold_detection_v5.py` | Main detection script (run this first) |
| `verify_holds.py` | Creates review images for manual checking |
| `hold_annotator.html` | Web-based tool for adding/removing holds |
| `holds_final.json` | Auto-detected holds (150 found) |
| `detected_holds_final.png` | Visualization of auto-detected holds |
| `holds_review.png` | High-res review image with numbered holds |

## Quick Start

### Step 1: Run Auto-Detection
```bash
python hold_detection_v5.py
```

This creates `holds_final.json` with detected hold coordinates.

### Step 2: Review & Correct

**Option A: Web Tool (Recommended)**
1. Open `hold_annotator.html` in your browser
2. Load your wall image
3. Load `holds_final.json`
4. Use Add/Remove modes to fix any errors
5. Export the corrected JSON

**Option B: Manual Review**
1. Open `holds_review.png`
2. Note any missing holds (write down pixel coordinates)
3. Note any false positives (hold IDs to remove)
4. Edit `holds_final.json` manually

### Step 3: Use in ML Pipeline

The output JSON has this structure:

```json
{
  "holds": [
    {
      "hold_id": 0,
      "pixel_x": 500,
      "pixel_y": 150,
      "norm_x": 0.27,      // 0-1, left to right
      "norm_y": 0.85,      // 0-1, BOTTOM to TOP
      "area": 1500,
      "color_rgb": [120, 180, 90],
      "confidence": 0.8
    },
    ...
  ]
}
```

## Coordinate System

```
norm_y = 1.0  ────────────────────  (top of wall)
              │                  │
              │    WALL AREA     │
              │                  │
norm_y = 0.0  ────────────────────  (bottom of wall)
           norm_x=0           norm_x=1
```

## Adding Additional Hold Features

After detection, you'll want to add more features for your ML model. Suggested workflow:

1. **Export to spreadsheet** (CSV) for manual labeling:
```python
import json
import csv

with open('holds_final.json') as f:
    data = json.load(f)

with open('holds_features.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['hold_id', 'norm_x', 'norm_y', 'type_hand', 'type_foot', 
                     'difficulty', 'angle_x', 'angle_y', 'hold_class'])
    for h in data['holds']:
        writer.writerow([h['hold_id'], h['norm_x'], h['norm_y'], 
                        '', '', '', '', '', ''])  # Fill these in manually
```

2. **Add features via annotation tool** - extend `hold_annotator.html` to include:
   - Dropdown for hold type (jug, crimp, sloper, pinch, pocket)
   - Checkbox for hand/foot usability
   - Slider for difficulty rating
   - Arrow tool for pulling direction

## Tips for Better Detection

1. **Lighting**: Even lighting across the wall helps
2. **Photo angle**: Shoot perpendicular to the wall face
3. **Contrast**: Works best when holds are colorful vs gray background
4. **Wall bounds**: Adjust `wall_bounds` in the script if detection includes non-wall areas

## Tunable Parameters

In `hold_detection_v5.py`, you can adjust:

```python
# In the detect_holds() call:
min_area=200,    # Minimum hold size in pixels² (increase to filter small noise)
max_area=30000   # Maximum hold size (decrease to filter large false positives)

# In the saturation mask:
s > 30           # Saturation threshold (lower = more sensitive)
v < 60           # Darkness threshold for black holds
v > 200 & s < 35 # Brightness threshold for white holds
```

## Troubleshooting

**Too few holds detected:**
- Lower the saturation threshold (e.g., `s > 20`)
- Check if wall bounds are correct
- Ensure image quality is good

**Too many false positives:**
- Increase `min_area`
- Tighten wall bounds
- Increase saturation threshold

**Holds merging together:**
- Decrease the `kernel_close` size
- Increase the DBSCAN `eps` parameter

## Next Steps

Once you have clean hold coordinates:

1. Add the manual features (difficulty, type, pulling direction)
2. Record climb sequences as described in your ML plan
3. Create position embeddings: `[hold_0_features, hold_1_features, ...]`
4. Train your sequence model!

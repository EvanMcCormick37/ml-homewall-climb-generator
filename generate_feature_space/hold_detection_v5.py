"""
Hold Detection v5 - Final Working Version
==========================================
Uses saturation mask with tight wall bounds.
"""

import cv2
import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


@dataclass
class DetectedHold:
    hold_id: int
    pixel_x: int
    pixel_y: int  
    norm_x: float
    norm_y: float
    area: int
    color_rgb: Tuple[int, int, int]
    confidence: float


def detect_holds(image_path: str, 
                 wall_bounds: Tuple[int, int, int, int],
                 min_area: int = 200,
                 max_area: int = 35000) -> List[DetectedHold]:
    """Detect holds using saturation-based segmentation."""
    
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    x1, y1, x2, y2 = wall_bounds
    wall = img[y1:y2, x1:x2]
    wall_rgb = img_rgb[y1:y2, x1:x2]
    h, w = wall.shape[:2]
    
    print(f"Cropped wall region: {w}x{h}")
    
    # HSV for saturation detection
    hsv = cv2.cvtColor(wall, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]  # Saturation
    v = hsv[:, :, 2]  # Value (brightness)
    
    # === Primary mask: Saturated (colorful) regions ===
    # Holds are colorful, plywood is gray (low saturation)
    mask_color = (s > 30).astype(np.uint8) * 255
    
    # === Secondary: Very dark regions (black holds) ===
    mask_dark = (v < 60).astype(np.uint8) * 255
    
    # === Tertiary: Very bright low-sat (white/cream holds) ===
    mask_white = ((v > 200) & (s < 35)).astype(np.uint8) * 255
    
    # Combine
    combined = cv2.bitwise_or(mask_color, mask_dark)
    combined = cv2.bitwise_or(combined, mask_white)
    
    # === Clean up ===
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Remove noise (small specks)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_open, iterations=1)
    # Fill small holes within holds
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    
    # Save debug
    debug_dir = Path("/home/claude/output/debug")
    debug_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(debug_dir / "v5_final_mask.png"), combined)
    
    # === Find contours ===
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Found {len(contours)} raw contours")
    
    holds = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Skip too small or too large
        if area < min_area or area > max_area:
            continue
        
        # Get centroid
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # Get average color
        mask_single = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask_single, [contour], -1, 255, -1)
        mean_color = cv2.mean(wall_rgb, mask=mask_single)[:3]
        
        # Confidence based on circularity
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
        else:
            circularity = 0
        confidence = min(1.0, circularity + 0.1)
        
        holds.append(DetectedHold(
            hold_id=-1,
            pixel_x=cx + x1,  # Convert back to full image coords
            pixel_y=cy + y1,
            norm_x=cx / w,
            norm_y=1.0 - (cy / h),  # Flip so 0=bottom, 1=top
            area=int(area),
            color_rgb=(int(mean_color[0]), int(mean_color[1]), int(mean_color[2])),
            confidence=confidence
        ))
    
    print(f"After area filter: {len(holds)} holds")
    
    # Deduplicate nearby detections
    if len(holds) > 1:
        from sklearn.cluster import DBSCAN
        coords = np.array([[h.pixel_x, h.pixel_y] for h in holds])
        clustering = DBSCAN(eps=30, min_samples=1).fit(coords)
        
        unique = []
        for label in set(clustering.labels_):
            cluster = [h for h, l in zip(holds, clustering.labels_) if l == label]
            # Keep largest in each cluster
            best = max(cluster, key=lambda x: x.area)
            unique.append(best)
        holds = unique
        print(f"After deduplication: {len(holds)} holds")
    
    # Sort top-to-bottom, left-to-right and assign IDs
    holds.sort(key=lambda h: (-h.norm_y, h.norm_x))
    for i, hold in enumerate(holds):
        hold.hold_id = i
    
    return holds


def visualize_results(image_path: str, holds: List[DetectedHold],
                      output_path: str, wall_bounds: Tuple[int, int, int, int]):
    """Create visualization with detected holds numbered."""
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    fig, ax = plt.subplots(figsize=(24, 18))
    ax.imshow(img_rgb)
    
    # Draw wall bounds
    x1, y1, x2, y2 = wall_bounds
    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                          fill=False, edgecolor='yellow', linewidth=2, linestyle='--')
    ax.add_patch(rect)
    
    # Draw holds
    for hold in holds:
        # Circle size based on area
        radius = max(10, np.sqrt(hold.area / np.pi))
        
        circle = Circle((hold.pixel_x, hold.pixel_y), radius,
                        fill=False, edgecolor='lime', linewidth=2)
        ax.add_patch(circle)
        
        # Number label
        ax.annotate(str(hold.hold_id),
                   (hold.pixel_x, hold.pixel_y),
                   color='white', fontsize=5, ha='center', va='center',
                   fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.12', facecolor='red', alpha=0.85))
    
    ax.set_title(f'Detected {len(holds)} holds', fontsize=18)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved: {output_path}")


def main():
    IMAGE_PATH = "/mnt/user-data/uploads/IMG_20250830_194551049__1_.jpg"
    OUTPUT_DIR = Path("/home/claude/output")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    img = cv2.imread(IMAGE_PATH)
    img_h, img_w = img.shape[:2]
    print(f"Full image: {img_w}x{img_h}")
    
    # === TIGHT wall bounds - excluding sky/trees/grass ===
    # Based on the image: wall is the central rectangular plywood area
    wall_bounds = (
        75,           # Left (inside the wood frame)
        70,           # Top (below the top trim and sky)
        1925,         # Right (inside the wood frame)
        1125          # Bottom (above the kickboard/grass)
    )
    
    print(f"Wall bounds: {wall_bounds}")
    
    # Detect
    holds = detect_holds(
        IMAGE_PATH,
        wall_bounds,
        min_area=200,     # Filter tiny noise
        max_area=30000    # Filter huge false positives
    )
    
    print(f"\n{'='*50}")
    print(f"FINAL RESULT: {len(holds)} holds detected")
    print(f"{'='*50}")
    
    # Visualize
    viz_path = OUTPUT_DIR / "detected_holds_final.png"
    visualize_results(IMAGE_PATH, holds, str(viz_path), wall_bounds)
    
    # Export JSON
    json_path = OUTPUT_DIR / "holds_final.json"
    data = {
        "metadata": {
            "wall_name": "Home Spray Wall",
            "image_path": IMAGE_PATH,
            "image_dimensions": [img_w, img_h],
            "wall_bounds_px": wall_bounds,
            "num_holds": len(holds),
            "coordinate_system": "norm_x: 0=left, 1=right; norm_y: 0=bottom, 1=top"
        },
        "holds": [asdict(h) for h in holds]
    }
    
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"JSON exported: {json_path}")
    
    # Summary statistics
    areas = [h.area for h in holds]
    print(f"\nHold statistics:")
    print(f"  Count: {len(holds)}")
    print(f"  Area range: {min(areas)} - {max(areas)} px²")
    print(f"  Mean area: {np.mean(areas):.0f} px²")
    
    # Sample output
    print(f"\nFirst 10 holds (top of wall):")
    for h in holds[:10]:
        print(f"  #{h.hold_id:3d}: pos=({h.norm_x:.3f}, {h.norm_y:.3f}) "
              f"area={h.area:5d} color=RGB{h.color_rgb}")
    
    return holds


if __name__ == "__main__":
    holds = main()

import os
from PIL import Image

def center_overlay_images(input_folder, output_filepath):
    """
    Reads transparent PNGs from a folder, creates a canvas large enough 
    to hold the biggest one, and overlays them all perfectly centered.
    """
    # Restricting to PNG since we are working with transparency
    valid_extensions = ('.png',) 
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)]
    files.sort() # Ensures they stack in alphabetical order
    
    if not files:
        print(f"No PNG images found in '{input_folder}'.")
        return

    print(f"Found {len(files)} images. Calculating canvas size...")

    # Step 1: Find the maximum width and height among all images
    max_width = 0
    max_height = 0
    images = []
    
    for file in files:
        img_path = os.path.join(input_folder, file)
        # Convert to RGBA to ensure the alpha channel (transparency) is active
        img = Image.open(img_path).convert("RGBA")
        images.append((file, img))
        
        if img.width > max_width:
            max_width = img.width
        if img.height > max_height:
            max_height = img.height

    print(f"Base canvas size will be {max_width}x{max_height} pixels.")

    # Step 2: Create a blank, fully transparent base canvas
    canvas = Image.new("RGBA", (max_width, max_height), (0, 0, 0, 0))

    # Step 3: Overlay each image in the center
    for file, img in images:
        print(f"Centering and overlaying {file}...")
        
        # Calculate exactly where to place the top-left corner of the image
        # so that it sits perfectly in the middle of the canvas
        x_offset = (max_width - img.width) // 2
        y_offset = (max_height - img.height) // 2
        
        # Paste the image using its own transparency as a mask
        canvas.paste(img, (x_offset, y_offset), img)

    # Step 4: Save the final image
    canvas.save(output_filepath, "PNG")
    print(f"\nSuccess! Saved centered stacked image to: {output_filepath}")

# --- Configuration ---
# Replace these strings with your actual folder path and desired output name
INPUT_DIRECTORY = "data/boardlib/kilter-images/full-ride" 
OUTPUT_FILE = "data/boardlib/kilter-images/full-ride.png"

# Run the script
if __name__ == "__main__":
    if not os.path.exists(INPUT_DIRECTORY):
        os.makedirs(INPUT_DIRECTORY)
        print(f"Created folder '{INPUT_DIRECTORY}'. Please put your transparent PNGs inside and run again.")
    else:
        center_overlay_images(INPUT_DIRECTORY, OUTPUT_FILE)
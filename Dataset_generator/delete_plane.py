import os
import json
import numpy as np
from PIL import Image
import ast

def clean_plane_segmentation(folder_path):
    """
    Iterates through a folder, finds PNG/JSON pairs, and removes pixels
    corresponding to 'Plane' in the segmentation map.
    """
    
    # ensure the folder path is valid
    if not os.path.exists(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")
        return

    # List all files in the directory
    files = os.listdir(folder_path)
    
    # Filter for JSON files to drive the process
    json_files = [f for f in files if f.endswith('.json')]

    print(f"Found {len(json_files)} JSON files. Processing...")

    for json_file in json_files:
        base_id = json_file[:10]
        png_file = base_id + ".png"
        
        json_path = os.path.join(folder_path, json_file)
        png_path = os.path.join(folder_path, png_file)

        # Check if the corresponding PNG exists
        if not os.path.exists(png_path):
            print(f"Skipping {json_file}: No corresponding {png_file} found.")
            continue

        try:
            # 1. Parse the JSON to find the "Plane" color
            with open(json_path, 'r') as f:
                data = json.load(f)

            plane_colors = []
            
            # Iterate through the dictionary to find keys containing "Plane"
            for color_key, label in data.items():
                if "Plane" in label:
                    # Convert string "(25, 255, 125, 255)" to tuple (25, 255, 125, 255)
                    try:
                        # ast.literal_eval is safer than eval() for parsing python-like structures
                        color_tuple = ast.literal_eval(color_key) 
                        plane_colors.append(color_tuple)
                    except (ValueError, SyntaxError):
                        print(f"Warning: Could not parse color key '{color_key}' in {json_file}")

            if not plane_colors:
                print(f"No 'Plane' labels found in {json_file}. Skipping.")
                continue

            # 2. Process the Image
            print(f"Processing {png_file}...")
            
            # Open image and convert to RGBA to ensure we handle transparency
            img = Image.open(png_path).convert("RGBA")
            data_array = np.array(img)

            # 3. Remove the colors
            # data_array is usually shape (Height, Width, 4)
            
            for color in plane_colors:
                target_color = np.array(color)
                
                # Create a mask: True where pixel matches target_color
                # We compare the first 3 or 4 channels depending on input
                # Assuming the JSON key includes Alpha, we compare all 4
                mask = np.all(data_array == target_color, axis=-1)
                
                # Set matching pixels to clear/transparent (0,0,0,0)
                data_array[mask] = [0, 0, 0, 0]

            # 4. Save the modified image
            # Overwrite the original file (or change path to save copy)
            new_img = Image.fromarray(data_array)
            new_img.save(png_path)
            print(f"Successfully cleaned {png_file}")

        except Exception as e:
            print(f"Error processing {base_name}: {e}")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    # Replace this with the actual path to your folder
    TARGET_FOLDER = "/home/robot/Workspace/Siemens_Cable_Simulator/output/capture_reload_each_test_2/seg" 
    
    clean_plane_segmentation(TARGET_FOLDER)
import os
import argparse
import shutil

def main():
    parser = argparse.ArgumentParser(description="Prepare dynamic model and rename images.")
    parser.add_argument("--input_dir", required=True, help="Directory containing the original .txt model")
    parser.add_argument("--output_dir", required=True, help="Directory to save the new filtered model")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Copy cameras.txt exactly as is
    cam_in = os.path.join(args.input_dir, "cameras.txt")
    cam_out = os.path.join(args.output_dir, "cameras.txt")
    if os.path.exists(cam_in):
        shutil.copy(cam_in, cam_out)

    # 2. Process images.txt
    img_in = os.path.join(args.input_dir, "images.txt")
    img_out = os.path.join(args.output_dir, "images.txt")
    
    with open(img_in, "r") as f_in, open(img_out, "w") as f_out:
        lines = f_in.readlines()
        i = 0
        
        while i < len(lines):
            line = lines[i]
            if line.startswith("#") or line.strip() == "":
                f_out.write(line)
                i += 1
                continue
                
            parts = line.strip().split()
            if len(parts) >= 10:
                original_name = " ".join(parts[9:])
                
                if "dynamic" in original_name:
                    # Extract the folder name (e.g., "dynamic1" from "dynamic1/0000.png")
                    # This assumes the word 'dynamic' is part of the parent folder name
                    dir_name = os.path.basename(os.path.dirname(original_name))
                    if not dir_name: # fallback if it's not in a subfolder
                        dir_name = os.path.splitext(original_name)[0]
                    
                    new_name = f"{dir_name}.png"
                    
                    # Reconstruct the line with the NEW image name
                    new_line = f"{parts[0]} {parts[1]} {parts[2]} {parts[3]} {parts[4]} {parts[5]} {parts[6]} {parts[7]} {parts[8]} {new_name}\n"
                    
                    f_out.write(new_line)
                    f_out.write("\n") # Blank line for points2D to force triangulation
                    
            i += 2 # Skip original 2D points line
            
    # 3. Create an empty points3D.txt
    open(os.path.join(args.output_dir, "points3D.txt"), "w").close()

if __name__ == "__main__":
    main()
import os
import shutil
import argparse
from pathlib import Path
import pycolmap

def filter_and_prepare_model(input_dir: Path, output_dir: Path, database_path: Path):
    """
    Filters the COLMAP txt model to only include dynamic images,
    renames them based on their directory, and clears 2D/3D points to force triangulation.

    The image and camera ids are remapped to the ones in the dynamic database
    (matched by image name) — the point triangulator looks images up by id, so
    the model must use the database's ids, not the static reconstruction's.
    The calibrated intrinsics from the static reconstruction are kept.
    """
    print("Filtering and formatting text model...")
    output_dir.mkdir(parents=True, exist_ok=True)

    db = pycolmap.Database.open(database_path)
    db_images = {im.name: im for im in db.read_all_images()}
    db.close()

    # 1. Parse cameras.txt: camera_id -> [MODEL, WIDTH, HEIGHT, PARAMS...]
    static_cameras = {}
    with open(input_dir / "cameras.txt", "r") as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue
            parts = line.strip().split()
            static_cameras[parts[0]] = parts[1:]

    # 2. Process images.txt
    img_in = input_dir / "images.txt"
    img_out = output_dir / "images.txt"
    db_camera_specs = {}  # db camera id -> calibrated camera spec

    with open(img_in, "r") as f_in, open(img_out, "w") as f_out:
        lines = f_in.readlines()
        i = 0

        while i < len(lines):
            line = lines[i]
            if line.startswith("#") or line.strip() == "":
                i += 1
                continue

            parts = line.strip().split()
            if len(parts) >= 10:
                original_name = " ".join(parts[9:])

                if "dynamic" in original_name:
                    # Extract the folder name
                    dir_name = Path(original_name).parent.name
                    if not dir_name:  # fallback if it's not in a subfolder
                        dir_name = Path(original_name).stem

                    new_name = f"{dir_name}.png"
                    db_image = db_images.get(new_name)

                    if db_image is None:
                        print(f"WARNING: '{new_name}' is not in the dynamic database, skipping.")
                    else:
                        # Give the database camera the intrinsics calibrated
                        # by the static reconstruction.
                        db_camera_specs.setdefault(db_image.camera_id, static_cameras[parts[8]])

                        # Reconstruct the line with the database ids and NEW image name
                        new_line_parts = [str(db_image.image_id)] + parts[1:8] + [str(db_image.camera_id), new_name]
                        f_out.write(" ".join(new_line_parts) + "\n")
                        f_out.write("\n") # Blank line for points2D to force triangulation

            i += 2 # Skip original 2D points line

    # 3. Write cameras.txt with the database camera ids
    with open(output_dir / "cameras.txt", "w") as f:
        for cam_id, spec in sorted(db_camera_specs.items()):
            f.write(" ".join([str(cam_id)] + spec) + "\n")

    # 4. Create an empty points3D.txt
    (output_dir / "points3D.txt").touch()


def main():
    parser = argparse.ArgumentParser(description="End-to-end COLMAP dynamic pipeline.")
    parser.add_argument("--data_dir", required=True, type=str, help="Path containing 'images' and 'dynamic-images' folders")
    parser.add_argument("--output_dir", required=True, type=str, help="Path to save the final triangulated model")
    parser.add_argument("--keep_temps", action="store_true", help="Pass this flag to prevent deleting intermediate files")
    args = parser.parse_args()

    # Define Base Paths
    data_dir = Path(args.data_dir)
    static_images_dir = data_dir / "images"
    dynamic_images_dir = data_dir / "dynamic-images"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Define Intermediate Paths
    db_static = out_dir / "database.db"
    db_dynamic = out_dir / "dynamic_database.db"
    distorted_dir = out_dir / "distorted"
    undistorted_dir = out_dir / "undistorted"
    sparse_text_dir = out_dir / "sparse_text"
    manual_sparse_dir = out_dir / "manual_sparse_dynamic"
    
    # Define Final Path
    final_output_dir = out_dir / "sparse_dynamic_triangulated"
    final_output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure directories exist
    distorted_dir.mkdir(exist_ok=True)
    undistorted_dir.mkdir(exist_ok=True)

    # --- PIPELINE START ---
    
    reader_options = pycolmap.ImageReaderOptions()
    reader_options.camera_model = "SIMPLE_RADIAL"

    # 1. Feature Extractor (Static)
    print("1/9: Extracting static features...")
    pycolmap.extract_features(
        database_path=db_static,
        image_path=static_images_dir,
        reader_options=reader_options
    )

    # 2. Exhaustive Matcher (Static)
    print("2/9: Matching static features...")
    pycolmap.match_exhaustive(database_path=db_static)

    # 3. Mapper
    print("3/9: Mapping...")
    reconstructions = pycolmap.incremental_mapping(
        database_path=db_static,
        image_path=static_images_dir,
        output_path=distorted_dir
    )
    if not reconstructions:
        raise RuntimeError("Mapping failed: no reconstruction could be built from the static images.")

    # 4. Image Undistorter
    print("4/9: Undistorting images...")
    # Pick the reconstruction with the most registered images
    best_idx = max(reconstructions, key=lambda idx: reconstructions[idx].num_reg_images())
    pycolmap.undistort_images(
        output_path=undistorted_dir,
        input_path=distorted_dir / str(best_idx),
        image_path=static_images_dir,
        output_type="COLMAP"
    )

    # 5. Feature Extractor (Dynamic)
    print("5/9: Extracting dynamic features...")
    pycolmap.extract_features(
        database_path=db_dynamic,
        image_path=dynamic_images_dir,
        reader_options=reader_options,
        camera_mode=pycolmap.CameraMode.SINGLE
    )

    # 6. Exhaustive Matcher (Dynamic)
    print("6/9: Matching dynamic features...")
    pycolmap.match_exhaustive(database_path=db_dynamic)

    # 7. Model Converter (Binary -> TXT)
    print("7/9: Converting binary model to text...")
    # Use the distorted model: its cameras (SIMPLE_RADIAL, original image
    # size) match the dynamic database, which was built from the raw dynamic
    # images. The undistorted model's PINHOLE cameras have cropped dimensions
    # and would be rejected during triangulation.
    sparse_text_dir.mkdir(exist_ok=True)
    reconstruction = pycolmap.Reconstruction(distorted_dir / str(best_idx))
    reconstruction.write_text(sparse_text_dir)

    # 8. Python Processing Logic
    print("8/9: Running internal text processing logic...")
    filter_and_prepare_model(input_dir=sparse_text_dir, output_dir=manual_sparse_dir, database_path=db_dynamic)

    # 9. Point Triangulator
    print("9/9: Triangulating points...")
    pycolmap.triangulate_points(
        reconstruction=pycolmap.Reconstruction(manual_sparse_dir),
        database_path=db_dynamic,
        image_path=dynamic_images_dir,
        output_path=final_output_dir
    )

    # --- CLEANUP ---
    if not args.keep_temps:
        print("Cleaning up intermediate files...")
        
        # Helper to silently delete directories
        def rm_dir(path: Path):
            if path.exists() and path.is_dir():
                shutil.rmtree(path)
                
        # Helper to silently delete files
        def rm_file(path: Path):
            if path.exists() and path.is_file():
                path.unlink()

        # Remove directories
        rm_dir(distorted_dir)
        rm_dir(undistorted_dir)
        rm_dir(sparse_text_dir)
        rm_dir(manual_sparse_dir)
        
        # Remove databases
        rm_file(db_static)
        rm_file(db_dynamic)
        
        print(f"Cleanup complete. Final model is located at: {final_output_dir}")
    else:
        print("Intermediate files kept intact.")

if __name__ == "__main__":
    main()
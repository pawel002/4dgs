#!/bin/bash

# Exit immediately if any command fails
set -e

if [ $# -ne 1 ]; then
    echo "Error: You must provide exactly one argument."
    echo "Usage: ./script.sh <path to input dir>"
    exit 1
fi

# Remove trailing slash from DIR if the user adds one by mistake
DIR="${1%/}"
DB_PATH="$DIR/database_bg.db"

echo "Creating directories..."
mkdir -p "$DIR/distorted"
mkdir -p "$DIR/temp_undistorted"
mkdir -p "$DIR/background"
mkdir -p "$DIR/dynamic"
mkdir -p "$DIR/dynamic-images"

# --------------------------------
# Pipeline for Static Images
# --------------------------------
echo "Starting static pipeline..."

# 1. Extract the features from images
colmap feature_extractor \
    --database_path "$DB_PATH" \
    --image_path "$DIR/images" \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model SIMPLE_RADIAL \
    --SiftExtraction.domain_size_pooling 1 \
    --SiftExtraction.max_num_features 2048

# 2. Feature matching
colmap exhaustive_matcher \
    --database_path "$DB_PATH"

# 3. Map points
colmap mapper \
    --database_path "$DB_PATH" \
    --image_path "$DIR/images" \
    --output_path "$DIR/distorted"

# 4. Undistort
colmap image_undistorter \
    --image_path "$DIR/images" \
    --input_path "$DIR/distorted/0" \
    --output_path "$DIR/temp_undistorted" \
    --output_type COLMAP

# Move the undistorted sparse model contents into background, then cleanup
mv "$DIR/temp_undistorted/sparse/"* "$DIR/background/"
rm -rf "$DIR/temp_undistorted"

# --------------------------------
# Pipeline for Dynamic Images
# --------------------------------
echo "Starting dynamic pipeline..."

DB_PATH_DYN="$DIR/database_dynamic.db"

# 1. Copy first frames from frames/dynamic*/ to $DIR/dynamic-images
for dir in "$DIR/frames/dynamic"*/; do
    [ -d "$dir" ] || continue
    
    files=("$dir"*)
    first_file="${files[0]}"
    
    if [ -f "$first_file" ]; then
        dir_name=$(basename "$dir")
        cp "$first_file" "$DIR/dynamic-images/${dir_name}.png"
    fi
done

# 2. Extract features from the NEW dynamic images into the NEW database
colmap feature_extractor \
    --database_path "$DB_PATH_DYN" \
    --image_path "$DIR/dynamic-images" \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model SIMPLE_RADIAL \
    --SiftExtraction.domain_size_pooling 1 \
    --SiftExtraction.max_num_features 2048

# 3. Match the features for the new dynamic images
colmap exhaustive_matcher \
    --database_path "$DB_PATH_DYN"

# 4. Convert the existing BACKGROUND binary model to text files (to get the camera poses)
mkdir -p "$DIR/sparse_text"
colmap model_converter \
    --input_path "$DIR/background" \
    --output_path "$DIR/sparse_text" \
    --output_type TXT

# 5. Filter for "dynamic" cameras and rename them to match dynamic-images folder
mkdir -p "$DIR/manual_sparse_dynamic"
python prepare_dynamic_model.py \
    --input_dir "$DIR/sparse_text" \
    --output_dir "$DIR/manual_sparse_dynamic"

# 6. Triangulate the points using the NEW database and the NEW images folder
mkdir -p "$DIR/sparse_dynamic_triangulated"
colmap point_triangulator \
    --database_path "$DB_PATH_DYN" \
    --image_path "$DIR/dynamic-images" \
    --input_path "$DIR/manual_sparse_dynamic" \
    --output_path "$DIR/sparse_dynamic_triangulated"

echo "Pipeline completed successfully!"
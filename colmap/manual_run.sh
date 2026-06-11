# greyed out lines work!

# colmap feature_extractor --database_path "./db/database.db" --image_path "./images" --ImageReader.camera_model SIMPLE_RADIAL

# colmap exhaustive_matcher --database_path "./db/database.db"

# colmap mapper --database_path "./db/database.db" --image_path "./images" --output_path "./distorted"

# colmap image_undistorter --image_path "./images" --input_path "./distorted/0" --output_path "./undistorted" --output_type COLMAP

# colmap feature_extractor --database_path "./db/dynamic_database.db" --image_path "./dynamic-images" --ImageReader.single_camera 1 --ImageReader.camera_model SIMPLE_RADIAL

# colmap exhaustive_matcher --database_path "./db/dynamic_database.db"

# colmap model_converter --input_path "./undistorted/sparse" --output_path "./sparse_text" --output_type TXT

# python prepare_dynamic_model.py --input_dir "./data/test1/sparse_text" --output_dir "./data/test1/manual_sparse_dynamic" 

colmap point_triangulator --database_path "./db/dynamic_database.db" --image_path "./dynamic-images" --input_path "./manual_sparse_dynamic" --output_path "./sparse_dynamic_triangulated"
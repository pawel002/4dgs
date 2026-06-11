# How to run colmap.

## Installation (using conda)

First you need COLMAP installed in conda env TODO.

## Running the reconstruction

This pipeline does the point cloud generation and pose estimation from mutiple pictures taken with the same camera.

### Step 1: Feature Extraction

```bash
colmap feature_extractor
    --database_path database.db
    --image_path {input dir}
    --ImageReader.single_camera 1
    --ImageReader.camera_model OPENCV
    --SiftExtraction.domain_size_pooling 1
    --SiftExtraction.max_num_features 2048
```

formatted:

```bash
colmap feature_extractor --database_path database.db --image_path test2/input --ImageReader.single_camera 1 --ImageReader.camera_model OPENCV --SiftExtraction.domain_size_pooling 1 --SiftExtraction.max_num_features 2048
```

what do the options mean:

- `--ImageReader.single_camera 1`: Tells COLMAP that all images were taken with the same camera, applying your parameters to every image.

- `--ImageReader.camera_model OPENCV`: PINHOLE assumes no lens distortion. If our lens has distortion that we haven't corrected for, we might want to use OPENCV or RADIAL.

### Step 2: Feature Matching

```bash
colmap exhaustive_matcher
    --database_path database.db
```

### Step 3: Mapping

We need output directory:

```bash
mkdir sparse
```

Then we can run the mapper:

```bash
colmap mapper
    --database_path database.db
    --image_path images
    --output_path sparse
```

```bash
colmap mapper --database_path database.db --image_path ./data/input --output_path ./data/output
```

### Step 4: Undistorting for 3DGS

Gaussian splatting needs pinhole type cameras, therefore we need to use undistorting algorithm from colmap to achieve this:

```bash
colmap image_undistorter --image_path ./data/input --input_path ./data/output/0 --output_path ./data/output/1 --output_type COLMAP
```

### Step 5: finding the positions of static camera views

I think that way better idea would generating actual point cloud with those static cameras so they will take part in the point cloud generation process, only after entire process they would be extracted. Perform feature extraction on new images:

```bash
colmap feature_extractor --database_path database.db --image_path ./test/input/ --image_list_path ./test/input_static.txt
```

Exhaustive matcher:

```bash
colmap exhaustive_matcher --database_path database.db
```

Get new static camera entrisics:

```bash
colmap image_registrator --database_path database.db --input_path ./data/output/0 --output_path ./data/output/1
```

Extracting only new cameras from those views:

```bash
colmap model_converter --input_path ./data/output/1 --output_path ./data/output/2 --output_type TXT
```

Now we can create a bash file that will perform everything on COLMAPS side. 

TODO:

- change options description
- describe what undistorter does

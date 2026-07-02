import bpy
import os
import json
import math

OUTPUT_DIR = "./output2"
CAMERA_NAMES = ["static1", "static2", "static3", "static4"]
# ==========================================

def enable_gpu_rendering():
    """Forces Blender to use Cycles and enables NVIDIA OptiX/CUDA GPUs."""
    print("Configuring GPU rendering...")
    
    # Set the render engine to Cycles
    bpy.context.scene.render.engine = 'CYCLES'
    
    # Tell the scene to use GPU compute
    bpy.context.scene.cycles.device = 'GPU'
    
    # Access the Cycles preferences
    cycles_prefs = bpy.context.preferences.addons['cycles'].preferences
    
    # Set the compute API to OptiX (Best for RTX/Ada generation cards)
    cycles_prefs.compute_device_type = 'OPTIX'
    
    # Refresh the device list so Blender actively probes the server hardware
    cycles_prefs.get_devices()
    
    # Loop through all detected hardware and ONLY enable the GPUs
    for device in cycles_prefs.devices:
        if device.type in ['OPTIX', 'CUDA']:
            device.use = True
            print(f"✅ Enabled GPU: {device.name}")
        else:
            device.use = False
            print(f"❌ Disabled CPU: {device.name}")

def get_camera_intrinsics(scene, cam):
    """Computes nerfstudio-compatible pinhole intrinsics (pixels) for a Blender camera.

    Returns a dict with fl_x, fl_y, cx, cy, w, h and the horizontal/vertical
    field-of-view angles. nerfstudio's `nerfstudio-data` dataparser requires the
    explicit focal lengths fl_x/fl_y and principal point cx/cy (camera_angle_x
    alone is not enough), so we emit all of them.
    """
    cam_data = cam.data
    f_mm = cam_data.lens
    sensor_w = cam_data.sensor_width
    sensor_h = cam_data.sensor_height

    # Effective rendered resolution (account for the resolution % slider).
    scale = scene.render.resolution_percentage / 100.0
    w = int(scene.render.resolution_x * scale)
    h = int(scene.render.resolution_y * scale)

    pa_x = scene.render.pixel_aspect_x
    pa_y = scene.render.pixel_aspect_y

    # Resolve AUTO sensor fit to the dimension Blender actually fits the sensor to.
    sensor_fit = cam_data.sensor_fit
    if sensor_fit == 'AUTO':
        sensor_fit = 'HORIZONTAL' if (w * pa_x >= h * pa_y) else 'VERTICAL'

    if sensor_fit == 'HORIZONTAL':
        fl_x = f_mm / sensor_w * w
        fl_y = fl_x * (pa_x / pa_y)
    else:  # VERTICAL
        fl_y = f_mm / sensor_h * h
        fl_x = fl_y * (pa_y / pa_x)

    return {
        "w": w,
        "h": h,
        "fl_x": fl_x,
        "fl_y": fl_y,
        "cx": w / 2.0,
        "cy": h / 2.0,
        "camera_angle_x": 2.0 * math.atan(w / (2.0 * fl_x)),
        "camera_angle_y": 2.0 * math.atan(h / (2.0 * fl_y)),
    }

def export_multi_camera_splatting():
    scene = bpy.context.scene
    
    # Save original state to restore later
    original_frame = scene.frame_current
    original_camera = scene.camera

    enable_gpu_rendering()

    print(f"Starting batch export to {OUTPUT_DIR}...\n")

    for cam_name in CAMERA_NAMES:
        # Find the camera object in the scene
        cam = scene.objects.get(cam_name)
        if not cam or cam.type != 'CAMERA':
            print(f"Warning: Camera '{cam_name}' not found or is not a camera. Skipping.")
            continue
            
        print(f"--- Processing Camera: {cam_name} ---")
        
        # Set this camera as the active rendering camera
        scene.camera = cam
        
        # Setup specific directories for this camera
        cam_dir = os.path.join(OUTPUT_DIR, cam_name)
        images_dir = os.path.join(cam_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        # Because the camera is static, we only need to calculate the pose matrix ONCE
        matrix = cam.matrix_world.normalized()
        static_transform_matrix = [list(row) for row in matrix]

        # Initialize the transforms dictionary with explicit nerfstudio intrinsics
        intr = get_camera_intrinsics(scene, cam)
        transforms_data = {
            "camera_angle_x": intr["camera_angle_x"],
            "camera_angle_y": intr["camera_angle_y"],
            "fl_x": intr["fl_x"],
            "fl_y": intr["fl_y"],
            "cx": intr["cx"],
            "cy": intr["cy"],
            "w": intr["w"],
            "h": intr["h"],
            "aabb_scale": 16,
            "frames": []
        }

        # Loop through the timeline and render the animation for THIS camera
        for frame in range(scene.frame_start, scene.frame_end + 1):
            scene.frame_set(frame)
            
            # Setup Filepath
            filename = f"{frame:04d}.png" 
            filepath = os.path.join(images_dir, filename)
            scene.render.filepath = filepath
            
            # Render the Frame
            print(f"Rendering {cam_name} - Frame {frame}...")
            bpy.ops.render.render(write_still=True)
            
            # Append to frames list (using the identical static matrix for every frame)
            transforms_data["frames"].append({
                "file_path": f"images/{filename}",
                "transform_matrix": static_transform_matrix
            })

        # Save the transforms.json file for this specific camera
        json_path = os.path.join(cam_dir, "transforms.json")
        with open(json_path, "w") as f:
            json.dump(transforms_data, f, indent=4)
            
        print(f"Finished {cam_name}! Data saved to {cam_dir}\n")

    # Restore scene to original state
    scene.frame_set(original_frame)
    scene.camera = original_camera
    print("All cameras processed successfully!")

# Run the function
export_multi_camera_splatting()
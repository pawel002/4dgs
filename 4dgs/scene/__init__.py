import json
import os
import random

from arguments import ModelParams
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from utils.camera_utils import camera_to_JSON, cameraList_from_camInfos
from utils.system_utils import searchForMaxIteration


class Scene:

    gaussians: GaussianModel

    def __init__(self, args: ModelParams, gaussians: GaussianModel,
                 load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(
                    os.path.join(self.model_path, "point_cloud")
                )
            else:
                self.loaded_iter = load_iteration
            print(f"Loading trained model at iteration {self.loaded_iter}")

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](
                args.source_path, args.images, args.depths, args.eval, args.train_test_exp
            )
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](
                args.source_path, args.white_background, args.depths, args.eval
            )
        else:
            raise RuntimeError("Could not recognize scene type!")

        if not self.loaded_iter:
            with open(scene_info.ply_path, "rb") as src, \
                 open(os.path.join(self.model_path, "input.ply"), "wb") as dst:
                dst.write(src.read())

            camlist = list(scene_info.test_cameras) + list(scene_info.train_cameras)
            json_cams = [camera_to_JSON(i, cam) for i, cam in enumerate(camlist)]
            with open(os.path.join(self.model_path, "cameras.json"), "w") as f:
                json.dump(json_cams, f)

        if shuffle:
            random.shuffle(scene_info.train_cameras)
            random.shuffle(scene_info.test_cameras)

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[scale] = cameraList_from_camInfos(
                scene_info.train_cameras, scale, args, scene_info.is_nerf_synthetic, False
            )
            print("Loading Test Cameras")
            self.test_cameras[scale] = cameraList_from_camInfos(
                scene_info.test_cameras, scale, args, scene_info.is_nerf_synthetic, True
            )

        if self.loaded_iter:
            self.gaussians.load_ply(
                os.path.join(self.model_path, "point_cloud",
                             f"iteration_{self.loaded_iter}", "point_cloud.ply"),
                args.train_test_exp,
            )
        else:
            self.gaussians.create_from_pcd(
                scene_info.point_cloud, scene_info.train_cameras, self.cameras_extent
            )

    def save(self, iteration):
        point_cloud_path = os.path.join(
            self.model_path, "point_cloud", f"iteration_{iteration}"
        )
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

        exposure_dict = {
            name: self.gaussians.get_exposure_from_name(name).detach().cpu().numpy().tolist()
            for name in self.gaussians.exposure_mapping
        }
        with open(os.path.join(self.model_path, "exposure.json"), "w") as f:
            json.dump(exposure_dict, f, indent=2)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

import json
import os

import numpy as np
import torch
from torch import nn
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2

from utils.general_utils import (
    build_rotation,
    build_scaling_rotation,
    get_expon_lr_func,
    inverse_sigmoid,
    strip_symmetric,
)
from utils.graphics_utils import BasicPointCloud
from utils.sh_utils import RGB2SH
from utils.system_utils import mkdir_p

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except ImportError:
    pass


class GaussianModel:

    def __init__(self, sh_degree: int, optimizer_type: str = "default"):
        self.max_sh_degree = sh_degree
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type

        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)

        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)

        self.optimizer = None
        self.exposure_optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0

        self.exposure_mapping = {}
        self.pretrained_exposures = None
        self._exposure = None

        self._setup_activations()

    def _setup_activations(self):
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

        def build_covariance(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            covariance = L @ L.transpose(1, 2)
            return strip_symmetric(covariance)

        self.covariance_activation = build_covariance

    # --- Properties ---

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        return torch.cat((self._features_dc, self._features_rest), dim=1)

    @property
    def get_features_dc(self):
        return self._features_dc

    @property
    def get_features_rest(self):
        return self._features_rest

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is not None:
            return self.pretrained_exposures[image_name]
        return self._exposure[self.exposure_mapping[image_name]]

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    @property
    def num_points(self):
        return self._xyz.shape[0]

    # --- SH degree ---

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    # --- Initialization ---

    def create_from_pcd(self, pcd: BasicPointCloud, cam_infos, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale

        points = torch.tensor(np.asarray(pcd.points), dtype=torch.float, device="cuda")

        if points.shape[0] == 0:
            num_random = 10_000
            print(f"Empty point cloud — initializing with {num_random} random points in some rectangle that should remind of the center!!! USE WITH CAUTION")
            mins = torch.tensor([0.45, 0, 1.35], device="cuda")
            maxs = torch.tensor([0.75, 1.8, 1.65], device="cuda")

            # Generate random values between 0 and 1, then scale them to your bounds
            points = torch.rand(num_random, 3, device="cuda") * (maxs - mins) + mins
            colors = torch.full((num_random, 3), 0.5, device="cuda")
            colors_sh = RGB2SH(colors)
        else:
            colors_sh = RGB2SH(torch.tensor(np.asarray(pcd.colors), dtype=torch.float, device="cuda"))

        num_points = points.shape[0]
        features = torch.zeros(num_points, 3, (self.max_sh_degree + 1) ** 2, dtype=torch.float, device="cuda")
        features[:, :3, 0] = colors_sh
        # Higher-order SH coefficients start at zero

        print(f"Number of points at initialization: {num_points}")

        dist2 = torch.clamp_min(distCUDA2(points), 1e-7)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rotations = torch.zeros(num_points, 4, device="cuda")
        rotations[:, 0] = 1.0
        opacities = inverse_sigmoid(0.1 * torch.ones(num_points, 1, dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(points.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rotations.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros(num_points, device="cuda")

        # Exposure: one 3x4 affine matrix per camera image
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    # --- Optimizer setup ---

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense

        self.xyz_gradient_accum = torch.zeros(self.num_points, 1, device="cuda")
        self.denom = torch.zeros(self.num_points, 1, device="cuda")

        param_groups = [
            {"params": [self._xyz], "lr": training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {"params": [self._features_dc], "lr": training_args.feature_lr, "name": "f_dc"},
            {"params": [self._features_rest], "lr": training_args.feature_lr / 20.0, "name": "f_rest"},
            {"params": [self._opacity], "lr": training_args.opacity_lr, "name": "opacity"},
            {"params": [self._scaling], "lr": training_args.scaling_lr, "name": "scaling"},
            {"params": [self._rotation], "lr": training_args.rotation_lr, "name": "rotation"},
        ]

        if self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(param_groups, lr=0.0, eps=1e-15)
            except Exception:
                self.optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)
        else:
            self.optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )
        self.exposure_scheduler_args = get_expon_lr_func(
            training_args.exposure_lr_init,
            training_args.exposure_lr_final,
            lr_delay_steps=training_args.exposure_lr_delay_steps,
            lr_delay_mult=training_args.exposure_lr_delay_mult,
            max_steps=training_args.iterations,
        )

    def update_learning_rate(self, iteration):
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group["lr"] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr

    # --- Checkpoint save/restore ---

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
        ) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    # --- PLY I/O ---

    def _construct_ply_attributes(self):
        attrs = ["x", "y", "z", "nx", "ny", "nz"]
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            attrs.append(f"f_dc_{i}")
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            attrs.append(f"f_rest_{i}")
        attrs.append("opacity")
        for i in range(self._scaling.shape[1]):
            attrs.append(f"scale_{i}")
        for i in range(self._rotation.shape[1]):
            attrs.append(f"rot_{i}")
        return attrs

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scales = self._scaling.detach().cpu().numpy()
        rotations = self._rotation.detach().cpu().numpy()

        dtype_full = [(attr, "f4") for attr in self._construct_ply_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        elements[:] = list(map(tuple, np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scales, rotations), axis=1
        )))

        PlyData([PlyElement.describe(elements, "vertex")]).write(path)

    def load_ply(self, path, use_train_test_exp=False):
        plydata = PlyData.read(path)

        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {
                    name: torch.FloatTensor(val).requires_grad_(False).cuda()
                    for name, val in exposures.items()
                }
                print("Pretrained exposures loaded.")
            else:
                print(f"No exposure file at {exposure_file}")
                self.pretrained_exposures = None

        verts = plydata.elements[0]
        xyz = np.stack((np.asarray(verts["x"]), np.asarray(verts["y"]), np.asarray(verts["z"])), axis=1)
        opacities = np.asarray(verts["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(verts["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(verts["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(verts["f_dc_2"])

        extra_f_names = sorted(
            [p.name for p in verts.properties if p.name.startswith("f_rest_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(verts[attr_name])
        features_extra = features_extra.reshape(xyz.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)

        scale_names = sorted(
            [p.name for p in verts.properties if p.name.startswith("scale_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(verts[attr_name])

        rot_names = sorted(
            [p.name for p in verts.properties if p.name.startswith("rot")],
            key=lambda x: int(x.split("_")[-1]),
        )
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(verts[attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    # --- Opacity reset ---

    def reset_opacity(self):
        new_opacity = self.inverse_opacity_activation(
            torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01)
        )
        optimizable_tensors = self._replace_tensor_in_optimizer(new_opacity, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    # --- Optimizer tensor manipulation ---

    def _replace_tensor_in_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
            optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            extension = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension)), dim=0)
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension), dim=0).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension), dim=0).requires_grad_(True)
                )
            optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    # --- Densification ---

    def prune_points(self, prune_mask):
        keep_mask = ~prune_mask
        optimizable_tensors = self._prune_optimizer(keep_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[keep_mask]
        self.denom = self.denom[keep_mask]
        self.max_radii2D = self.max_radii2D[keep_mask]

    def _densification_postfix(self, new_xyz, new_features_dc, new_features_rest,
                               new_opacities, new_scaling, new_rotation):
        tensors = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }
        optimizable_tensors = self._cat_tensors_to_optimizer(tensors)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros(self.num_points, 1, device="cuda")
        self.denom = torch.zeros(self.num_points, 1, device="cuda")
        self.max_radii2D = torch.zeros(self.num_points, device="cuda")

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        selected = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected = torch.logical_and(
            selected,
            torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent,
        )

        self._densification_postfix(
            self._xyz[selected],
            self._features_dc[selected],
            self._features_rest[selected],
            self._opacity[selected],
            self._scaling[selected],
            self._rotation[selected],
        )

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init = self.num_points
        padded_grad = torch.zeros(n_init, device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()

        selected = torch.where(padded_grad >= grad_threshold, True, False)
        selected = torch.logical_and(
            selected,
            torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent,
        )

        stds = self.get_scaling[selected].repeat(N, 1)
        means = torch.zeros(stds.size(0), 3, device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected]).repeat(N, 1, 1)

        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected].repeat(N, 1)
        new_features_dc = self._features_dc[selected].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected].repeat(N, 1, 1)
        new_opacity = self._opacity[selected].repeat(N, 1)

        self._densification_postfix(
            new_xyz, new_features_dc, new_features_rest,
            new_opacity, new_scaling, new_rotation,
        )

        prune_filter = torch.cat((
            selected,
            torch.zeros(N * selected.sum(), device="cuda", dtype=bool),
        ))
        self.prune_points(prune_filter)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = prune_mask | big_points_vs | big_points_ws
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1

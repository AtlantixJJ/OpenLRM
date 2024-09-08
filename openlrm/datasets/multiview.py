import os
import json
import math
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F

from PIL import Image
from torchvision.transforms import ToTensor, Compose, Resize, Lambda, CenterCrop
from os.path import join as osj
from .cam_utils import build_camera_standard, build_camera_principle, camera_normalization_objaverse, center_looking_at_camera_pose


def imread_pil(fpath, size=None):
    img = Image.open(open(fpath, "rb")).convert("RGB")
    if size is not None:
        img = img.resize(size, resample=Image.Resampling.BILINEAR)
    return img


def farthest_point_sampling(points, num_samples,
                            replace=False,
                            dist_mode='mean',
                            sampled_indices=None):
    """
    Performs farthest point sampling on a set of points.
    Low efficiency, do not use when num_samples is large.
    Args:
        points: A NumPy array of shape (num_points, dimension) representing the points.
        num_samples: The number of points to sample.
        sampled_indices: A list of indices that have already been sampled.
        dist_mode: 'mean' or 'min', the distance of new point to existing points.
        replace: Whether to sample with replacement (repeated points are allowed).
    Returns:
        sampled indices
    """

    # Check input validity
    if not replace and num_samples > len(points):
        raise ValueError("Number of samples cannot be greater than the number of points when replace=False!")

    N = points.shape[0]
    indices = np.arange(N)

    reduce_fn = lambda x: np.min(x, axis=-1)
    if dist_mode == 'mean':
        reduce_fn = lambda x: np.mean(x, axis=-1)

    # Randomly choose the first point
    if sampled_indices is None:
        sampled_indices = [np.random.randint(0, N, size=1)[0]]

    # Calculate distances to the sampled points
    diff = points[:, None] - points[sampled_indices][None]
    dist2 = reduce_fn((diff ** 2).sum(-1))
    p = dist2 / dist2.sum().clip(min=1e-6)

    # Iteratively select farthest points
    for _ in range(len(sampled_indices), num_samples):
        sampled_idx = int(np.random.choice(indices, size=1, p=p))
        while not replace and sampled_idx in sampled_indices:
            sampled_idx = int(np.random.choice(indices, size=1, p=p))

        # Add the farthest point to the sampled set
        sampled_indices.append(sampled_idx)

        diff = points[:, None] - points[sampled_indices][None]
        dist2 = reduce_fn((diff ** 2).sum(-1))
        p = dist2 / dist2.sum()

    return sampled_indices


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def intrinsics_from_fov(
        fov_x: float,
        fov_y: float):
    """Get the camera intrinsics matrix from FoV.
    Args:
        fov: Field of View in radian.
    """
    fx = 1 / np.tan(fov_y / 2)
    fy = 1 / np.tan(fov_x / 2)

    return torch.Tensor([
        [fx, 0, 0.5],
        [0, -fy, 0.5],
        [0, 0, 1]
    ])


class MultiViewDataset(torch.utils.data.Dataset):
    """Multi-view dataset.
    """

    def __init__(self, root_dirs: str, meta_path: str, use_depth=False,
                 max_size=-1, sample_acam=1,
                 sample_side_views=3,
                 render_image_res_low: int = 64,
                 render_image_res_high: int = 192,
                 render_region_size: int = 64,
                 source_image_res: int = 224,
                 normalize_camera: bool = True,
                 normed_dist_to_center: str = 'auto'
                ):
        """
        Args:
            root_dir: str, root directory of the dataset.
            ann_file: str, annotation file.
            max_size: int, maximum number of samples.
            sample_acam: float, sample around camera, quaternion threshold.
            resolution: int, resolution of the images.
        """
        self.meta_path = meta_path
        self.root_dir = root_dirs[0]
        self.render_image_res_low = render_image_res_low
        self.render_image_res_high = render_image_res_high
        self.render_region_size = render_region_size
        self.source_image_res = source_image_res
        self.normalize_camera = normalize_camera
        self.normed_dist_to_center = normed_dist_to_center
        self.n_ref = 1
        self.n_tarfar = sample_side_views
        self.sample_side_views = sample_side_views
        self.n_tarnear = 0
        self.use_depth = use_depth
        self.sample_acam = sample_acam # sample around camera, L2 threshold
        self.scene_df = pd.read_parquet(meta_path)
        self.max_size = max_size
        capture_names = self.scene_df['capture'].unique()
        if self.max_size > 0:
            capture_names = capture_names[:self.max_size]
        self.scene2idx = {name: idx for idx, name in enumerate(capture_names)}
        #self.idx2scene = {idx: name for name, idx in self.scene2idx.items()}
        self.transforms = ToTensor()

    def __len__(self):
        return len(self.scene_df)

    def get_camera_dict(self, scene_prefix, scene_dic, cam_idx):
        """Get camera dictionary."""

        image_path = osj(scene_prefix, scene_dic['image_path'][cam_idx])
        mask_path = osj(scene_prefix, scene_dic['mask_path'][cam_idx])
        image = self.transforms(imread_pil(image_path))
        mask = self.transforms(imread_pil(mask_path))[:1]
        width, height = image.shape[1:]

        if self.use_depth:
            depth_path = osj(scene_prefix, scene_dic['depth_path'][cam_idx])
            depth = np.asarray(Image.open(open(depth_path, "rb")))
            # need to use scale invariant loss
            depth_min, depth_max = 2.1, 3.3 # this number is verified
            depth = depth / 65535.0 * (depth_max - depth_min) + depth_min
            depth = self.transforms(Image.fromarray(depth))
        else:
            depth = torch.zeros_like(mask)

        return dict(
            image=image, mask=mask, depth=depth,
            image_width=width, image_height=height,
            image_path=image_path)

    def choose_cameras(self, cam_pos):
        n_cam = cam_pos.shape[0]
        if self.n_ref == -1:
            ref_cam_indices = list(range(n_cam))
        else:
            ref_cam_indices = farthest_point_sampling(cam_pos, self.n_ref)

        # find near target cameras
        diff = cam_pos[:, None] - cam_pos[ref_cam_indices][None]
        dist = np.sqrt((diff ** 2).sum(-1))
        dist[dist < 1e-3] = 1e6 # remove self
        if self.n_tarnear == -1:
            tarnear_cam_indices = list(range(n_cam))
        elif self.n_tarnear > 0:
            tarnear_cam_indices = [0] * self.n_tarnear
            for i in range(self.n_tarnear):
                ref_idx = i % self.n_ref
                #near_inds = dist[ref_idx] < dist[ref_idx].min() * (1 + self.sample_acam)
                near_inds = dist[:, ref_idx] < self.sample_acam

                near_inds = near_inds.nonzero()[0]
                if near_inds.shape[0] == 0:
                    near_inds = list(range(dist.reshape(-1).shape[0]))
                tarnear_cam_indices[i] = int(np.random.choice(near_inds))
        else:
            tarnear_cam_indices = []

        # other target cameras are used for union gaussian inference
        if self.n_tarfar == -1:
            tarfar_cam_indices = list(range(n_cam))
        elif self.n_tarfar > 0:
            tarfar_cam_indices = np.random.choice(
                np.arange(n_cam), self.n_tarfar,
                replace=n_cam < self.n_tarfar)
        else:
            tarfar_cam_indices = []
        return ref_cam_indices, tarnear_cam_indices, tarfar_cam_indices

    def __getitem__(self, idx):
        """
        Returns:
            tar_camera: n_views cameras. Each corresponding to the closest ref camera.
            ref_camera: n_views cameras. 
        """
        # notice: we randomize here to avoid resetting every epoch
        #idx = np.random.randint(len(self.sample_indice))
        #scene_idx, tar_cam_idx = self.sample_indice[idx]

        scene_df = self.scene_df.iloc[int(idx)]
        scene_name = scene_df['capture']
        scene_prefix = osj(self.root_dir, scene_name)
        if not os.path.exists(scene_prefix):
            scene_prefix = self.root_dir

        world2cam = scene_df['world2cam'].reshape(-1, 4, 4)
        cam2world = np.linalg.inv(world2cam) # R, T
        cam_pos = cam2world[:, :3, 3]

        ref_cam_indices, tarnear_cam_indices, tarfar_cam_indices = \
            self.choose_cameras(cam_pos)
        #ref_cam_indices = [0]
        #tarfar_cam_indices = [1, 2, 19, 20, 40, 41, 70, 71]

        ref_cameras = [self.get_camera_dict(scene_prefix, scene_df, cam_idx)
                       for cam_idx in ref_cam_indices]
        tarfar_cameras = [self.get_camera_dict(scene_prefix, scene_df, cam_idx)
                       for cam_idx in tarfar_cam_indices]
        all_cams = ref_cameras[:1] + tarfar_cameras
        all_indices = list(ref_cam_indices)[:1] + list(tarfar_cam_indices)

        """
        # RT (3, 4) cat with fx fy cx cy -> 12 + 4 = 16
        def get_principle_camera_matrix(dic):
            RT = dic['cam2world'][:3, :4].reshape(-1)
            K = dic['intrinsics']
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            return torch.cat([RT, torch.stack([fx, fy, cx, cy])])
    
        def get_standard_camera_matrix(dic):
            E = dic['cam2world'].reshape(-1)
            I = dic['intrinsics'].reshape(-1)
            return torch.cat([E, I])

        source_camera = get_principle_camera_matrix(ref_cameras[0])
        render_camera = torch.stack([get_standard_camera_matrix(c)
                                     for c in all_cams])
        """
        R = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]]).astype('float32')
        cam_pos = cam_pos[all_indices] @ R.T
        poses = center_looking_at_camera_pose(torch.from_numpy(cam_pos))
        intrinsic = torch.tensor([[0.75, 0.75], [0.5, 0.5], [1.0, 1.0]])

        #poses = torch.stack([c['cam2world'] for c in all_cams])[:, :3, :4]
        #intrinsics = torch.stack([c['intrinsics'] for c in all_cams])
        if self.normalize_camera:
            poses = camera_normalization_objaverse(self.normed_dist_to_center, poses)
        # build source and target camera features
        
        source_camera = build_camera_principle(poses[:1], intrinsic.unsqueeze(0)).squeeze(0)
        render_camera = build_camera_standard(poses, intrinsic.repeat(poses.shape[0], 1, 1))

        source_image = ref_cameras[0]['image']
        render_image = torch.stack([c['image'] for c in all_cams])
        n_render = render_image.shape[0]
        full_size = source_image.shape[-1]

        resize_fn = lambda x, s : torch.nn.functional.interpolate(
            x, size=(s, s), mode='bicubic', align_corners=True)
        
        # for input processing of DINO
        source_image = resize_fn(source_image[None], self.source_image_res)[0]

        # for rendering just a part of the image
        render_image_res = np.random.randint(self.render_image_res_low, self.render_image_res_high + 1)
        render_image = resize_fn(render_image, render_image_res)
        
        anchors = torch.randint(
            0, render_image_res - self.render_region_size + 1, size=(n_render, 2))
        crop_indices = torch.arange(0, self.render_region_size, device=render_image.device)
        index_i = (anchors[:, 0].unsqueeze(1) + crop_indices).view(-1, self.render_region_size, 1)
        index_j = (anchors[:, 1].unsqueeze(1) + crop_indices).view(-1, 1, self.render_region_size)
        batch_indices = torch.arange(n_render, device=render_image.device).view(-1, 1, 1)

        cropped_render_image = render_image[batch_indices, :, index_i, index_j].permute(0, 3, 1, 2)

        full_resolutions = torch.tensor([[render_image_res]], dtype=torch.float32).repeat(n_render, 1)
        bg_color = torch.ones(n_render, 1).float()

        return {
            'uid': self.scene2idx[scene_name],
            'source_camera': source_camera,
            'render_camera': render_camera,
            'source_image': source_image,
            'render_image': cropped_render_image,
            'render_anchors': anchors,
            'render_full_resolutions': full_resolutions,
            'render_bg_colors': bg_color,
        }

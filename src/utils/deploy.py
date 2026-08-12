# -*- encoding: utf-8 -*-
"""
@File    :   deploy.py
@Time    :   2025/05/09 10:43:18
@Author  :   yxing
"""

import cv2
import logging
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union

import torch
import torchvision.transforms.functional as TF

from config import Config


@dataclass
class DeploymentInfo:
    """Precomputed deployment information from Steps 1-4.
    
    Contains all geometric and tracking data needed to render a patch
    into stereo frames, without the actual rendering.
    """
    H_patch_to_left: np.ndarray           # (3, 3) homography from patch to left start frame
    H_patch_to_right: np.ndarray          # (3, 3) homography from patch to right start frame
    all_homographies_left: List[Optional[np.ndarray]]   # per-frame patch→left-frame homographies
    all_homographies_right: List[Optional[np.ndarray]]  # per-frame patch→right-frame homographies
    all_masks: List[np.ndarray]           # per-frame warped surface masks
    all_visibility: List[bool]            # per-frame visibility flags
    patch_corners_3d_left: np.ndarray     # (4, 3) 3D corners in left camera coords (at ref frame)
    patch_corners_3d_right: np.ndarray    # (4, 3) 3D corners in right camera coords (at ref frame)
    ref_frame_idx: int                    # frame index where 3D corners were computed
    start_idx: int                        # first visible frame index
    end_idx: int                          # last visible frame index


class DigitalDeploy():

    def __init__(self, dataset, calib_file, img_left, img_right):
        
        self.calib_file = calib_file
        if dataset == 'kitti':
            self.mk = ['P_rect_02', 'P_rect_03', 'R_rect_00']
        elif dataset == 'drivingstereo':
            self.mk = ['P_rect_101', 'P_rect_103', 'R_rect_101']
        else:
            raise ValueError("unknown dataset")

    def project(self, P, R0, corner):
        """corner: homogeneous coordinates of the board"""
        _m = np.matmul(P, R0)
        y = np.dot(_m, corner)

        y[0] /= y[2]
        y[1] /= y[2]
        y = y[:2]
        return tuple(y)

    def get_patch_spec(self, physical_size, physical_depth, center_shift=(0,0)):
        P0, P1, R = self.load_calibrition()

        x_shift, y_shift = center_shift
        board_height, board_width = physical_size

        imgL_tl=self.project(P0,R,[-board_width/2+x_shift,-board_height/2+y_shift,physical_depth,1])
        imgL_tr=self.project(P0,R,[ board_width/2+x_shift,-board_height/2+y_shift,physical_depth,1])
        imgL_bl=self.project(P0,R,[-board_width/2+x_shift, board_height/2+y_shift,physical_depth,1])
        imgL_br=self.project(P0,R,[ board_width/2+x_shift, board_height/2+y_shift,physical_depth,1])
        # print('Patch Corners in Left:')
        # print(imgL_tl,imgL_tr)
        # print(imgL_bl,imgL_br)
        # print('--------------------')
        imgR_tl=self.project(P1,R,[-board_width/2+x_shift,-board_height/2+y_shift,physical_depth,1])
        imgR_tr=self.project(P1,R,[ board_width/2+x_shift,-board_height/2+y_shift,physical_depth,1])
        imgR_bl=self.project(P1,R,[-board_width/2+x_shift, board_height/2+y_shift,physical_depth,1])
        imgR_br=self.project(P1,R,[ board_width/2+x_shift, board_height/2+y_shift,physical_depth,1])
        # print('Patch Corners in Right:')
        # print(imgR_tl,imgR_tr)
        # print(imgR_bl,imgR_br)
        # print('--------------------')

        # NOTE the discrimination of pixel location is very limited where only half of a pixel error may exists, thus we naively get the patch size
        lw = imgL_tr[0] - imgL_tl[0]
        lh = imgL_bl[1] - imgL_tl[1]
        # lcenter = (int(imgL_tl[0]+lw/2), int(imgL_tl[1]+lh/2))
        # print(lw, lh)

        rw = imgR_tr[0] - imgR_tl[0]
        rh = imgR_bl[1] - imgR_tl[1]
        # rcenter = (int(imgR_tl[0]+rw/2), int(imgR_tl[1]+rh/2))
        # print(rw, rh)

        return (imgL_tl, imgL_tr, imgL_bl, imgL_br), (imgR_tl, imgR_tr, imgR_bl, imgR_br), (lh, lw), (rh, rw)

    def load_calibrition(self):
        with open(self.calib_file, 'r') as f:
            calib_results = f.readlines()

        calib_dict = dict()
        for m in calib_results:
            _tmp = m.strip().split(' ')
            k, v = _tmp[0].split(':')[0], _tmp[1:]
            calib_dict[k] = v

        mx_required = []
        for k in self.mk:
            assert k in calib_dict.keys(), 'the required matrix not found in the loaded calibration file'

            mat = np.array([float(i) for i in calib_dict[k]])
            if 'R' in k:
                mat = np.pad(mat.reshape((3,3)), [(0, 1), (0, 1)], mode='constant')
                mat[-1,-1] = 1.0
            else:
                mat = mat.reshape((3,4))
            mx_required.append(mat)
        
        return mx_required
        
    def deploy(self, board, left_img, right_img, left_pos, right_pos, board_size, img_range=None, tau=0.5):
        """Embed the board into stereo images.
        Args:
            board (torch.Tensor): The board to be embedded.
            left_img (torch.Tensor): The left stereo image.
            right_img (torch.Tensor): The right stereo image.
            left_pos (tuple): List of positions for the left image.
            right_pos (tuple): List of positions for the right image.
            board_size (tuple): Size of the board.
        Returns:
            left_img (torch.Tensor): Left image with the board embedded.
            right_img (torch.Tensor): Right image with the board embedded.
        """
        # convert board into the image value range
        # _board = torch.clip(board, 0, 1)
        # _board = torch.sigmoid(board)
        # _board = (board - board.min()) / (board.max() - board.min() + 1e-6)
        # _board = torch.sigmoid(board-board.mean())
        # _mask= SignMaskSTE.apply(board)
        # _board = board * _mask
        # _board = (torch.tanh(board)+1)/2
        # _board = MinMaxNormalizeSTE.apply(board)
        # print(_board.min(), _board.max())
        _board = board
        
        left_tl_xy, right_tl_xy = left_pos[0], right_pos[0]

        # assemble the board first
        left_img[...,left_tl_xy[1]:left_tl_xy[1]+board_size[0],left_tl_xy[0]:left_tl_xy[0]+board_size[1]] = _board
        right_img[...,right_tl_xy[1]:right_tl_xy[1]+board_size[0],right_tl_xy[0]:right_tl_xy[0]+board_size[1]] = _board
        return left_img, right_img


class StereoPatchDeployer:
    def __init__(self, sceneloader: str):
        """Initialize with stereo calibration.
        
        Args:
            sceneloader: dataloader for Virtual KITTI 2 dataset
        """
        self.dataloader = sceneloader
        self.n_frames = len(sceneloader)
        self.cam_paras = sceneloader.camera_params

        self.K_left = self.cam_paras["Camera_0"]["intrinsic"]
        self.K_right = self.cam_paras["Camera_1"]["intrinsic"]
        self.R = self.cam_paras["stereo"]["R_left_to_right"]
        self.T = self.cam_paras["stereo"]["t_left_to_right"]
        self.P_left = self.cam_paras["Camera_0"]["projection_rect"]
        self.P_right = self.cam_paras["Camera_1"]["projection_rect"]

        self.baseline = self.cam_paras['stereo']['baseline']
    
    def back_project(self, u: float, v: float, depth: float) -> np.ndarray:
        """Back-project 2D point + depth to 3D (left camera coordinates)."""
        fx, fy = self.K_left[0, 0], self.K_left[1, 1]
        cx, cy = self.K_left[0, 2], self.K_left[1, 2]
        
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth
        
        return np.array([x, y, z])
    
    def project_3d_to_left(self, point_3d: np.ndarray) -> Tuple[float, float]:
        """Project 3D point to left image.
        
        Args:
            point_3d: 3D point in left camera coordinates (shape: (3,))
            
        Returns:
            (u, v): Pixel coordinates in left image
        """
        point_homo = self.P_left @ np.append(point_3d, 1)
        u = point_homo[0] / point_homo[2]
        v = point_homo[1] / point_homo[2]
        return u, v
    
    def project_3d_to_right(self, point_3d: np.ndarray) -> Tuple[float, float]:
        """Project 3D point (in left camera coordinates) to right image.
        
        P_right (projection_rect) already encodes the stereo baseline shift,
        so it directly maps left-camera 3D points to right-image pixels.
        No explicit left→right coordinate transform is needed.
        """
        point_homo = self.P_right @ np.append(point_3d, 1)
        u = point_homo[0] / point_homo[2]
        v = point_homo[1] / point_homo[2]
        return u, v
    
    def fit_plane_ransac(self, points_3d: np.ndarray, 
                        max_iterations: int = 1000,
                        inlier_threshold: float = None) -> Tuple[np.ndarray, float]:
        """Fit plane to 3D points using RANSAC."""
        
        # Auto-scale threshold based on depth
        if inlier_threshold is None:
            avg_depth = np.mean(points_3d[:, 2])  # Average Z
            inlier_threshold = max(0.02, avg_depth * 0.002)  # 0.2% of depth, min 2cm
            
        best_normal = None
        best_d = 0
        best_inliers = 0
        
        for _ in range(max_iterations):
            idx = np.random.choice(len(points_3d), 3, replace=False)
            sample = points_3d[idx]
            
            v1 = sample[1] - sample[0]
            v2 = sample[2] - sample[0]
            normal = np.cross(v1, v2)
            
            if np.linalg.norm(normal) < 1e-6:
                continue
                
            normal = normal / np.linalg.norm(normal)
            d = -np.dot(normal, sample[0])
            
            distances = np.abs(np.dot(points_3d, normal) + d)
            inlier_count = np.sum(distances < inlier_threshold)
            
            if inlier_count > best_inliers:
                best_inliers = inlier_count
                best_normal = normal
                best_d = d
        
        # Refit using inliers
        distances = np.abs(np.dot(points_3d, best_normal) + best_d)
        inliers = points_3d[distances < inlier_threshold]
        
        if len(inliers) > 3:
            centroid = np.mean(inliers, axis=0)
            centered = inliers - centroid
            U, S, Vt = np.linalg.svd(centered)
            normal = Vt[2]
            
            if normal[2] > 0:
                normal = -normal
            d = -np.dot(normal, centroid)
        else:
            normal, d = best_normal, best_d
        
        return normal, d
    
    def compute_uv_basis(self, points_3d: np.ndarray, 
                        plane_normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute UV basis vectors on the plane, aligned with camera image axes.
        
        u_axis corresponds to the camera's horizontal (X) direction projected onto the plane.
        v_axis corresponds to the camera's vertical (Y) direction projected onto the plane.
        This ensures patches appear upright in the camera view.
        """
        # Camera X-axis = [1,0,0], Y-axis = [0,1,0] in left camera coordinates.
        # Project onto the plane by removing the component along the plane normal.
        cam_x = np.array([1.0, 0.0, 0.0])
        cam_y = np.array([0.0, 1.0, 0.0])
        
        u_axis = cam_x - np.dot(cam_x, plane_normal) * plane_normal
        
        if np.linalg.norm(u_axis) < 1e-6:
            # Plane normal is nearly parallel to camera X — fall back to camera Y for u
            u_axis = cam_y - np.dot(cam_y, plane_normal) * plane_normal
        u_axis = u_axis / np.linalg.norm(u_axis)
        
        # v_axis: project camera Y onto the plane, then orthogonalize against u_axis
        v_axis = cam_y - np.dot(cam_y, plane_normal) * plane_normal
        # Remove any component along u_axis to guarantee orthogonality
        v_axis = v_axis - np.dot(v_axis, u_axis) * u_axis
        
        if np.linalg.norm(v_axis) < 1e-6:
            # Degenerate case: use cross product as fallback
            v_axis = np.cross(plane_normal, u_axis)
        v_axis = v_axis / np.linalg.norm(v_axis)
        
        return u_axis, v_axis
    
    def warp_points_with_flow(self, points: np.ndarray, 
                             flow_field: np.ndarray) -> np.ndarray:
        """Warp 2D points using ground truth optical flow.
        
        Args:
            points: Array of 2D points with shape (N, 2) where each row is [x, y]
            flow_field: Optical flow field with shape (h, w, 2) where flow_field[y, x] = [dx, dy]
                       Both dx and dy are in pixel units
        
        Returns:
            Warped points with shape (N, 2), same as input
            
        Implementation Details:
        - For each point, look up optical flow at the nearest integer pixel location
        - Add flow displacement to the original point coordinate
        - Maintains sub-pixel accuracy by preserving fractional coordinates
        - Points outside the flow field bounds are kept unchanged
        """
        warped_points = []
        h, w = flow_field.shape[:2]
        
        for x, y in points:
            # Get integer coordinates for flow lookup
            xi, yi = int(np.round(x)), int(np.round(y))
            
            # Check if the rounded coordinates are within bounds
            if 0 <= xi < w and 0 <= yi < h:
                # Look up optical flow at the nearest integer pixel location
                dx, dy = flow_field[yi, xi]
                
                # Add flow to original point (preserves sub-pixel precision)
                warped_points.append([x + dx, y + dy])
            else:
                # Point is outside bounds - keep unchanged
                warped_points.append([x, y])
        
        return np.array(warped_points)
    
    def warp_mask_with_flow(self, mask: np.ndarray, 
                           flow_field: np.ndarray) -> np.ndarray:
        """Warp binary mask using optical flow.
        
        Uses sub-pixel accuracy with strict boundary checking.
        NO dilation is applied - only pixels that actually project into valid regions are kept.
        
        Args:
            mask: Binary mask to warp
            flow_field: Optical flow field
            
        Returns:
            Warped mask with strict boundary validation
        """
        h, w = mask.shape
        warped_mask = np.zeros_like(mask, dtype=np.float32)
        
        # For each pixel in the original mask
        mask_indices = np.where(mask > 0.1)
        
        for y, x in zip(mask_indices[0], mask_indices[1]):
            if 0 <= x < w and 0 <= y < h:
                dx, dy = flow_field[y, x]
                new_x = x + dx
                new_y = y + dy
                
                # Use nearest neighbor with sub-pixel accumulation
                # This preserves pixels that might be lost by strict rounding
                x_floor, y_floor = int(np.floor(new_x)), int(np.floor(new_y))
                x_ceil, y_ceil = int(np.ceil(new_x)), int(np.ceil(new_y))
                
                # Bilinear interpolation weights
                wx = new_x - x_floor
                wy = new_y - y_floor
                
                # Distribute pixel contribution to 4 neighbors
                # STRICT: Only accumulate if target is within bounds
                for xi, weight_x in [(x_floor, 1 - wx), (x_ceil, wx)]:
                    for yi, weight_y in [(y_floor, 1 - wy), (y_ceil, wy)]:
                        # Strict boundary check: reject any out-of-bounds pixels
                        if 0 <= xi < w and 0 <= yi < h:
                            warped_mask[yi, xi] += weight_x * weight_y
        
        # Convert to binary with threshold
        # Accumulated weights from properly warped pixels are >= 0.25 (corner pixel contributes to 4 neighbors)
        # Scattered noise from out-of-bounds regions has very small accumulated weight
        #-TODO too sensitive here, 0.25 will make the surface still visible when it is actually disappeared, consider change the mask tracking methodology
        binary_mask = (warped_mask > 0.5).astype(mask.dtype)
        
        return binary_mask
    
    def track_surface_mask_with_flow(self,
                                    start_frame_idx: int,
                                    surface_mask: np.ndarray,
                                    direction: str = 'forward') -> Tuple[List[np.ndarray], List[bool]]:
        """Track surface mask across frames using optical flow.
        
        Returns:
            Tuple of (list of masks, list of visibility flags)
            - masks: Warped masks for each frame
            - visibility: Whether mask has non-zero area in each frame
        """
        masks = [surface_mask.copy()]
        visibility = [True]  # Start frame always has visibility
        
        if direction == 'forward':
            frame_idx_range = range(start_frame_idx + 1, self.n_frames)
        else:  # backward
            frame_idx_range = range(start_frame_idx - 1, 0 - 1, -1)
        
        current_mask = surface_mask.copy()
        
        for i in frame_idx_range:
            # Get optical flow for this frame
            # Forward: To go from frame i-1 to frame i, use flow_forward from frame i-1
            #         (flow_forward[i-1] = motion from i-1 to i)
            # Backward: To go from frame i+1 to frame i, use flow_backward from frame i+1
            #          (flow_backward[i+1] = motion from i+1 to i, per VirtualKITTI2 convention)
            #          Convention: rgb(t) + flow_backward(t) = rgb(t-1)
            
            flow_idx = i - 1 if direction == 'forward' else i + 1
            
            if 0 <= flow_idx < self.n_frames:
                frame_data = self.dataloader.get_frame(flow_idx)
                
                try:
                    optical_flow = frame_data['flow_forward' if direction == 'forward' else 'flow_backward']
                    if optical_flow is None:
                        raise KeyError("Flow data is None")
                except (KeyError, TypeError):
                    # Flow not available, mark as not visible and stop tracking
                    visibility.append(False)
                    masks.append(np.zeros_like(current_mask))
                    # Once mask becomes invisible, keep it invisible for subsequent frames
                    # (can't reliably track a disappeared object)
                    continue
                
                # Warp mask using optical flow
                prev_mask_area = np.sum(current_mask)
                current_mask = self.warp_mask_with_flow(current_mask, optical_flow)
                curr_mask_area = np.sum(current_mask)
                
                # Check if mask is still visible (has meaningful area, not just noise)
                # Threshold: 300 pixels minimum to consider mask as visible
                # This prevents tracking noise/artifacts when surface has left camera view
                min_visible_area = 300
                is_visible = curr_mask_area >= min_visible_area
                visibility.append(is_visible)
                masks.append(current_mask.copy())
                
                # Debug: log mask disappearance
                if prev_mask_area >= min_visible_area and curr_mask_area < min_visible_area:
                    print(f"\t  [DEBUG] {direction.upper()} Frame {i}: mask too small (was {prev_mask_area} pixels, now {curr_mask_area})")
        
        return masks, visibility
    
    def compute_homography_from_flow(self, prev_points: np.ndarray,
                                    flow_field: np.ndarray) -> np.ndarray:
        """Compute homography using optical flow correspondences."""
        if len(prev_points) < 4:
            return np.eye(3)
        
        curr_points = self.warp_points_with_flow(prev_points, flow_field)
        
        # Filter invalid points
        valid_mask = np.all(np.isfinite(curr_points), axis=1)
        prev_points = prev_points[valid_mask]
        curr_points = curr_points[valid_mask]
        
        if len(prev_points) < 4:
            return np.eye(3)
        
        H, _ = cv2.findHomography(
            prev_points.reshape(-1, 1, 2),
            curr_points.reshape(-1, 1, 2),
            cv2.RANSAC,
            ransacReprojThreshold=2.0
        )
        
        return H if H is not None else np.eye(3)
    
    def track_surface_with_flow(self, 
                               start_frame_idx: int,
                               surface_mask: np.ndarray,
                               direction: str = 'forward',
                               camera: int = 0) -> List[np.ndarray]:
        """Track surface across frames using optical flow.
        
        Returns list of homographies:
        - For forward: [H(start→start), H(start→start+1), H(start→start+2), ..., H(start→start+n)]
        - For backward: [H(start→start), H(start→start-1), H(start→start-2), ..., H(start→start-n)]
        where H(A→B) = homography that transforms points from frame A to frame B
        """
        homographies = []
        H_accumulated = np.eye(3)
        homographies.append(H_accumulated.copy())
        
        # Sample points from surface region
        h, w = surface_mask.shape
        mask_indices = np.where(surface_mask > 0)
        
        if len(mask_indices[0]) > 0:
            n_samples = min(1000, len(mask_indices[0]))
            sample_idx = np.random.choice(len(mask_indices[0]), n_samples, replace=False)
            points = np.column_stack([
                mask_indices[1][sample_idx],  # x
                mask_indices[0][sample_idx]   # y
            ])
        else:
            points = np.array([])
        
        if direction == 'forward':
            # Start from next frame (start_frame_idx + 1) and track forward
            frame_idx_range = range(start_frame_idx + 1, self.n_frames)
        else: # backward
            # Start from previous frame (start_frame_idx - 1) and track backward
            # Always stop at frame 0 (index -1 in range to include it)
            frame_idx_range = range(start_frame_idx - 1, 0 - 1, -1)
        
        prev_points = points.copy()
        
        for i in frame_idx_range:
            # For both directions: flow_idx is the frame FROM which flow originates
            # Forward: flow from i-1 → i, so flow_idx = i-1
            #          flow_forward[i-1] gives motion from frame i-1 to frame i
            # Backward: flow from i+1 → i, so flow_idx = i+1
            #          flow_backward[i+1] gives motion from frame i+1 to frame i
            #          Per VirtualKITTI2 convention: rgb(t) + flow_backward(t) = rgb(t-1)
            flow_idx = i - 1 if direction == 'forward' else i + 1
            
            if 0 <= flow_idx < self.n_frames:
                frame_data = self.dataloader.get_frame(flow_idx, camera=camera)
                try:
                    optical_flow = frame_data['flow_forward'] if direction == 'forward' else frame_data['flow_backward']
                except KeyError:
                    print(f"  [ERROR] Optical flow data missing for frame {flow_idx}. Available keys: {list(frame_data.keys())}")
                
                if len(prev_points) > 0:
                    H_step = self.compute_homography_from_flow(prev_points, optical_flow)
                    
                    # For both forward and backward, H_step maps from previous frame to current frame
                    # H_accumulated should map from start_frame to current frame
                    # So we compose: H_accumulated(new) = H_step @ H_accumulated(old)
                    H_accumulated = H_step @ H_accumulated
                    
                    prev_points = self.warp_points_with_flow(prev_points, optical_flow)
                
                homographies.append(H_accumulated.copy())
        
        return homographies
    
    def compute_right_depth_from_left(self, depth_left: np.ndarray,
                                     flow_left_to_right: np.ndarray = None) -> np.ndarray:
        """Compute right depth map from left depth and stereo geometry.
        
        Args:
            depth_left: Depth map for left view
            flow_left_to_right: Optional flow from left to right view
                               (if available for per-pixel correspondence)
        
        Returns:
            Depth map for right view
        """
        h, w = depth_left.shape
        
        if flow_left_to_right is not None:
            # Use flow for accurate per-pixel mapping
            depth_right = np.zeros_like(depth_left)
            
            for y in range(h):
                for x in range(w):
                    depth = depth_left[y, x]
                    if depth > 0:
                        dx, dy = flow_left_to_right[y, x]
                        x_right = x + dx
                        y_right = y + dy
                        
                        if 0 <= x_right < w and 0 <= y_right < h:
                            depth_right[int(y_right), int(x_right)] = depth
        else:
            # Approximate using stereo geometry
            # For rectified stereo: depth_right(x,y) ≈ depth_left(x + disparity,y)
            # Since we have ground truth depth, we can back-project and re-project
            depth_right = np.zeros_like(depth_left)
            
            for y in range(h):
                for x in range(w):
                    depth = depth_left[y, x]
                    if depth > 0:
                        pt_3d = self.back_project(x, y, depth)
                        pt_3d_right = self.R @ pt_3d + self.T.flatten()
                        depth_right[y, x] = pt_3d_right[2]  # z in right camera
        
        return depth_right
    
    def render_patch(self, frame: np.ndarray, depth_map: np.ndarray,
                    patch_img: np.ndarray, H_patch_to_frame: np.ndarray,
                    patch_corners_3d: np.ndarray) -> np.ndarray:
        """Render patch with occlusion handling and partial visibility support.
        
        Uses cropped canvas to prevent scan-line artifacts from full-frame warpPerspective.
        """
        h, w = frame.shape[:2]
        patch_h, patch_w = patch_img.shape[:2]
        
        # Check if homography is valid (all values are finite and reasonable)
        if not np.all(np.isfinite(H_patch_to_frame)):
            # Invalid homography, return frame unchanged
            return frame.copy()
        
        # CRITICAL: Check if patch corners project within image bounds
        # This ensures the patch is actually visible in the current frame
        patch_corners_pixel = np.array([
            [0, 0], [patch_w, 0], [patch_w, patch_h], [0, patch_h]
        ], dtype=np.float32)
        
        # Transform patch corners to frame space using homography
        corners_homo = (H_patch_to_frame @ np.vstack([patch_corners_pixel.T, np.ones(4)]))
        
        # Check for invalid homography (negative or zero z-coordinates indicate flipped geometry)
        if np.any(corners_homo[2, :] <= 0):
            # Patch is behind camera or has degenerate geometry
            return frame.copy()
        
        corners_frame = corners_homo[:2, :] / corners_homo[2, :]  # Perspective division
        
        # Check if ANY corner falls within valid image bounds
        # Allow tolerance for sub-pixel rendering: [-0.5, w+0.5] and [-0.5, h+0.5]
        corners_valid = (corners_frame[0, :] >= -0.5) & \
                        (corners_frame[0, :] <= w + 0.5) & \
                        (corners_frame[1, :] >= -0.5) & \
                        (corners_frame[1, :] <= h + 0.5)
        
        if not np.any(corners_valid):
            # All corners are completely outside image bounds
            return frame.copy()
        
        # Additional check: ensure at least a meaningful amount of patch is inside bounds
        # If no corners are actually inside (only near bounds), check if visible area is substantial
        corners_strictly_inside = (corners_frame[0, :] >= 0) & \
                                  (corners_frame[0, :] < w) & \
                                  (corners_frame[1, :] >= 0) & \
                                  (corners_frame[1, :] < h)
        
        # No corners strictly inside image - patch is at edge or outside
        if not np.any(corners_strictly_inside):
            # Check if bounding box has significant overlap
            bbox_width = np.max(corners_frame[0, :]) - np.min(corners_frame[0, :])
            bbox_height = np.max(corners_frame[1, :]) - np.min(corners_frame[1, :])
            bbox_area = bbox_width * bbox_height
            
            # If bounding box area is very small, likely a ghost patch at frame edge
            if bbox_area < 100:  # Minimum 100 square pixels
                return frame.copy()
        
        # Compute bounding box of projected patch corners
        patch_min_x = np.floor(np.min(corners_frame[0, :])).astype(int)
        patch_max_x = np.ceil(np.max(corners_frame[0, :])).astype(int)
        patch_min_y = np.floor(np.min(corners_frame[1, :])).astype(int)
        patch_max_y = np.ceil(np.max(corners_frame[1, :])).astype(int)
        
        # Add margin for interpolation artifacts (1-2 pixels)
        margin = 2
        patch_min_x = max(0, patch_min_x - margin)
        patch_max_x = min(w, patch_max_x + margin)
        patch_min_y = max(0, patch_min_y - margin)
        patch_max_y = min(h, patch_max_y + margin)
        
        patch_width = patch_max_x - patch_min_x
        patch_height = patch_max_y - patch_min_y
        
        if patch_width <= 0 or patch_height <= 0:
            return frame.copy()
        
        # Create translation homography to account for cropped canvas
        # Transforms from full frame coordinates to cropped canvas coordinates
        H_crop = np.array([[1, 0, -patch_min_x],
                          [0, 1, -patch_min_y],
                          [0, 0, 1]], dtype=np.float32)
        
        # Adjust homography for cropped output
        H_patch_to_crop = H_crop @ H_patch_to_frame
        
        #-error here is the part that requires careful modification
        # Warp patch to cropped canvas (NOT full frame)
        warped_patch = cv2.warpPerspective(
            patch_img, H_patch_to_crop, (patch_width, patch_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )
        
        # Ensure warped_patch has 4 channels (RGBA) for consistent processing
        if warped_patch.ndim == 2 or (warped_patch.ndim == 3 and warped_patch.shape[2] != 4):
            if warped_patch.ndim == 2:
                warped_patch = cv2.cvtColor(warped_patch, cv2.COLOR_GRAY2BGRA)
            elif warped_patch.shape[2] == 3:
                warped_patch = cv2.cvtColor(warped_patch, cv2.COLOR_BGR2BGRA)
        
        # Create patch alpha mask - use threshold to filter noise and interpolation artifacts
        # Alpha threshold: pixels with alpha <= 50/255 (~20%) are considered noise/artifacts
        # This preserves semi-transparent features while rejecting faint interpolation noise
        # Tunable: increase for stricter filtering (less feathering), decrease for softer edges
        alpha_threshold = 50  # Range: 0-255. Recommended: 20-80 (8%-31%)
        patch_alpha = warped_patch[:, :, 3] > alpha_threshold
        
        # If no visible pixels, return frame unchanged
        if not np.any(patch_alpha):
            return frame.copy()
        
        # Compute per-pixel depth using perspective interpolation from corner depth
        crop_h, crop_w = warped_patch.shape[:2]
        depth_warped = np.zeros((crop_h, crop_w), dtype=np.float32)
        
        # Create coordinate grids for the cropped warped space
        y_grid, x_grid = np.meshgrid(np.arange(crop_h), np.arange(crop_w), indexing='ij')
        
        # Adjust coordinates back to full frame coordinates for inverse homography
        x_full = x_grid.astype(np.float32) + patch_min_x
        y_full = y_grid.astype(np.float32) + patch_min_y
        
        # Back-project to patch space using inverse homography
        # Use conditioning check to avoid numerical instability
        cond = np.linalg.cond(H_patch_to_frame)
        if cond > 1e10:
            print(f"Warning: Homography is ill-conditioned (cond={cond:.2e}), skipping frame")
            return frame.copy()
        
        H_inv = np.linalg.inv(H_patch_to_frame)
        ones = np.ones((crop_h, crop_w, 1), dtype=np.float32)
        coords_full = np.concatenate([x_full[..., np.newaxis], y_full[..., np.newaxis], ones], axis=2)
        coords_patch = np.dot(coords_full, H_inv.T)
        
        # Normalize by homogeneous coordinate (perspective division)
        z_patch = coords_patch[..., 2] + 1e-8
        u_patch = coords_patch[..., 0] / z_patch
        v_patch = coords_patch[..., 1] / z_patch
        
        # Identify valid patch region (pixels that actually come from patch)
        valid_patch_pixels = (u_patch >= 0) & (u_patch < patch_w) & \
                            (v_patch >= 0) & (v_patch < patch_h)
        
        # Only compute depth for valid patch pixels
        u_patch_norm = np.zeros_like(u_patch)
        v_patch_norm = np.zeros_like(v_patch)
        
        u_patch_norm[valid_patch_pixels] = u_patch[valid_patch_pixels] / patch_w
        v_patch_norm[valid_patch_pixels] = v_patch[valid_patch_pixels] / patch_h
        
        # Interpolate depth from corners at each pixel
        corner_depths = patch_corners_3d[:, 2]  # [tl, tr, br, bl]
        depth_tl = corner_depths[0]
        depth_tr = corner_depths[1]
        depth_br = corner_depths[2]
        depth_bl = corner_depths[3]
        
        # Bilinear interpolation of depth (only for valid pixels)
        depth_top = (1 - u_patch_norm) * depth_tl + u_patch_norm * depth_tr
        depth_bottom = (1 - u_patch_norm) * depth_bl + u_patch_norm * depth_br
        depth_warped = (1 - v_patch_norm) * depth_top + v_patch_norm * depth_bottom
        
        # Mark invalid pixels as far away to ensure they don't blend
        depth_warped[~valid_patch_pixels] = np.inf
        
        # Prepare full frame result
        blended = frame.copy()
        
        # Extract cropped depth and frame regions
        depth_crop = depth_map[patch_min_y:patch_max_y, patch_min_x:patch_max_x]
        frame_crop = blended[patch_min_y:patch_max_y, patch_min_x:patch_max_x, :3]
        
        # Vectorized occlusion handling
        valid_depth = ~np.isnan(depth_crop) & (depth_crop > 0)
        occlusion_mask = patch_alpha.copy()
        
        # Pixels are occluded if patch depth is significantly farther than scene depth
        occlusion_threshold = 0.05  # 5cm threshold
        occlusion_mask[valid_depth & (depth_warped > depth_crop + occlusion_threshold)] = False
        occlusion_mask[~valid_depth | (depth_crop <= 0)] = False  # Reject invalid depths
        
        # Extract alpha channel and normalize to [0, 1]
        alpha = warped_patch[:, :, 3:4].astype(np.float32) / 255.0
        
        # Apply occlusion mask to alpha
        alpha[~occlusion_mask] = 0
        
        # Create blend mask: only blend where alpha > 0
        blend_mask = alpha[:, :, 0] > 0
        
        if np.any(blend_mask):
            # Extract RGB channels and ensure float32 for blending
            patch_rgb = warped_patch[:, :, :3].astype(np.float32)
            
            # Perform alpha blending with proper normalization
            blended_rgb = (1 - alpha[blend_mask]) * frame_crop[blend_mask].astype(np.float32) + \
                         alpha[blend_mask] * patch_rgb[blend_mask]
            
            # Clip values to valid range and convert back to original dtype
            blended_rgb = np.clip(blended_rgb, 0, 255).astype(frame.dtype)
            blended[patch_min_y:patch_max_y, patch_min_x:patch_max_x, :3][blend_mask] = blended_rgb

        return blended

    def render_patch_torch(self, frame: np.ndarray, depth_map: np.ndarray,
                          patch_img: torch.Tensor, H_patch_to_frame: np.ndarray,
                          patch_corners_3d: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
        """Render patch with occlusion handling using pure PyTorch operations.
        
        This is a torch-compatible version of render_patch that preserves computational graphs for gradient-based optimization.
        
        Args:
            frame: Input frame as numpy uint8 array (H, W, 3) in RGB format
            depth_map: Depth map as numpy float32 array (H, W)
            patch_img: Patch image as torch.Tensor (3, H, W) RGB in float32 [0, 1] range. Can be on CPU or GPU. Should have requires_grad=True for backprop.
            H_patch_to_frame: Homography matrix (3, 3) from patch to frame coordinates, numpy float64 array
            patch_corners_3d: corners' 3D positions (4, 3) for depth interpolation, numpy float64 array
        
        Returns:
            Tuple of:
            - Rendered frame as torch.Tensor (3, H, W) RGB in float32 [0, 1] range. Output is on the same device as patch_img and preserves computational graph.
            - Deployment mask as np.ndarray (H, W) bool, True where the patch is visibly rendered.
        
        Key differences from numpy render_patch:
        - Uses torch.nn.functional.grid_sample instead of cv2.warpPerspective
        - All operations preserve gradients for backpropagation
        - Returns float32 tensor in [0, 1] range instead of uint8
        - Bilinear interpolation fully differentiable
        """
        device = patch_img.device
        dtype = patch_img.dtype
        
        h, w = frame.shape[:2]
        _, patch_h, patch_w = patch_img.shape  # patch_img is (3, H, W)
        empty_mask = np.zeros((h, w), dtype=bool)
        
        # Convert inputs to torch tensors on the same device
        frame_t = torch.from_numpy(frame.astype(np.float32) / 255.0).to(device).permute(2, 0, 1)  # (3, H, W)
        depth_map_t = torch.from_numpy(depth_map.astype(np.float32)).to(device)  # (H, W)
        H_patch_to_frame_t = torch.from_numpy(H_patch_to_frame.astype(np.float64)).to(device)
        patch_corners_3d_t = torch.from_numpy(patch_corners_3d.astype(np.float32)).to(device)
        
        # ===== STEP 1: Validate homography and patch visibility =====
        if not torch.all(torch.isfinite(H_patch_to_frame_t)):
            return frame_t, empty_mask
        
        # Check if patch corners project within image bounds
        patch_corners_pixel = torch.tensor([
            [0.0, 0.0], [patch_w, 0.0], [patch_w, patch_h], [0.0, patch_h]
        ], dtype=torch.float64, device=device)  # (4, 2)
        
        # Add homogeneous coordinate
        ones = torch.ones(4, 1, dtype=torch.float64, device=device)
        patch_corners_homo_coords = torch.cat([patch_corners_pixel, ones], dim=1)  # (4, 3)
        
        # Transform to frame space: (4, 3) = (3, 3) @ (3, 4)
        corners_homo = (H_patch_to_frame_t @ patch_corners_homo_coords.T).T
        
        # Check for invalid homography
        if torch.any(corners_homo[:, 2] <= 0):
            return frame_t, empty_mask
        
        # Perspective division
        corners_frame = corners_homo[:, :2] / corners_homo[:, 2:3]  # (4, 2)
        
        # Reject if NO corner falls within valid image bounds (with sub-pixel tolerance)
        corners_valid = \
            (corners_frame[:, 0] >= -0.5) & (corners_frame[:, 0] <= w + 0.5) & \
            (corners_frame[:, 1] >= -0.5) & (corners_frame[:, 1] <= h + 0.5)
        
        if not torch.any(corners_valid):
            return frame_t, empty_mask
        
        # Check bounding box area
        corners_strictly_inside = \
            (corners_frame[:, 0] >= 0) & (corners_frame[:, 0] < w) & \
            (corners_frame[:, 1] >= 0) & (corners_frame[:, 1] < h)
        
        if not torch.any(corners_strictly_inside):
            bbox_width = torch.max(corners_frame[:, 0]) - torch.min(corners_frame[:, 0])
            bbox_height = torch.max(corners_frame[:, 1]) - torch.min(corners_frame[:, 1])
            bbox_area = bbox_width * bbox_height
            
            if bbox_area < 100:
                return frame_t, empty_mask
        
        # ===== STEP 2: Compute bounding box and cropped canvas =====
        patch_min_x = torch.floor(torch.min(corners_frame[:, 0])).long().item()
        patch_max_x = torch.ceil(torch.max(corners_frame[:, 0])).long().item()
        patch_min_y = torch.floor(torch.min(corners_frame[:, 1])).long().item()
        patch_max_y = torch.ceil(torch.max(corners_frame[:, 1])).long().item()
        
        # Add margin for interpolation artifacts
        margin = 2
        patch_min_x = max(0, patch_min_x - margin)
        patch_max_x = min(w, patch_max_x + margin)
        patch_min_y = max(0, patch_min_y - margin)
        patch_max_y = min(h, patch_max_y + margin)
        
        patch_width = patch_max_x - patch_min_x
        patch_height = patch_max_y - patch_min_y
        
        if patch_width <= 0 or patch_height <= 0:
            return frame_t, empty_mask
        
        # ===== STEP 3: Warp patch using grid_sample (differentiable) =====
        # Compute inverse homography in float64 for numerical precision, then cast grid to float32 for grid_sample
        cond = torch.linalg.cond(H_patch_to_frame_t)
        if cond > 1e10:
            print(f"Warning: Homography is ill-conditioned (cond={cond:.2e}), skipping frame")
            return frame_t, empty_mask
        
        H_inv = torch.linalg.inv(H_patch_to_frame_t)  # float64
        
        # Create coordinate grids for the cropped canvas (float64 for precise inverse mapping)
        y_out = torch.arange(patch_height, dtype=torch.float64, device=device)
        x_out = torch.arange(patch_width, dtype=torch.float64, device=device)
        y_grid, x_grid = torch.meshgrid(y_out, x_out, indexing='ij')  # (H_crop, W_crop)
        
        # Convert to full frame coordinates
        x_full = x_grid + patch_min_x
        y_full = y_grid + patch_min_y
        
        # Back-project to patch space using inverse homography
        ones_grid = torch.ones_like(x_full)
        coords_full = torch.stack([x_full, y_full, ones_grid], dim=-1)  # (H_crop, W_crop, 3)
        coords_patch = torch.matmul(coords_full, H_inv.T)  # (H_crop, W_crop, 3)
        
        # Perspective division
        z_patch = coords_patch[..., 2] + 1e-8
        u_patch = coords_patch[..., 0] / z_patch  # (H_crop, W_crop)
        v_patch = coords_patch[..., 1] / z_patch  # (H_crop, W_crop)
        
        # Identify valid patch region
        valid_patch_pixels = (u_patch >= 0) & (u_patch < patch_w) & \
                            (v_patch >= 0) & (v_patch < patch_h)
        
        # Normalize to [-1, 1] for grid_sample (align_corners=True)
        grid_x = 2.0 * u_patch / (patch_w - 1) - 1.0
        grid_y = 2.0 * v_patch / (patch_h - 1) - 1.0
        
        # Out-of-bounds pixels get zero-padded by grid_sample
        grid_x = torch.where(valid_patch_pixels, grid_x, torch.tensor(2.0, dtype=torch.float64, device=device))
        grid_y = torch.where(valid_patch_pixels, grid_y, torch.tensor(2.0, dtype=torch.float64, device=device))
        
        # Cast to float32 for grid_sample (which requires matching dtype with input tensor)
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).to(dtype)  # (1, H_crop, W_crop, 2)
        
        # Warp patch using grid_sample (fully differentiable)
        warped_patch = torch.nn.functional.grid_sample(
            patch_img.unsqueeze(0),  # (1, 3, H_patch, W_patch)
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        ).squeeze(0)  # (3, H_crop, W_crop)
        
        # ===== STEP 4: Check warped patch visibility =====
        # valid_patch_pixels (from Step 3) marks which crop pixels map to the patch interior.
        # Early exit if no valid pixels exist.
        if not torch.any(valid_patch_pixels):
            return frame_t, empty_mask
        
        # ===== STEP 5: Compute per-pixel depth interpolation =====
        # u_patch, v_patch, valid_patch_pixels already computed in STEP 3
        
        # Normalize to [0, 1] range for depth interpolation
        u_patch_norm = torch.clamp(u_patch / patch_w, 0.0, 1.0)
        v_patch_norm = torch.clamp(v_patch / patch_h, 0.0, 1.0)
        
        # Interpolate depth from 4 corners
        corner_depths = patch_corners_3d_t[:, 2]  # [tl, tr, br, bl]
        depth_tl = corner_depths[0]
        depth_tr = corner_depths[1]
        depth_br = corner_depths[2]
        depth_bl = corner_depths[3]
        
        # Bilinear interpolation (fully differentiable)
        depth_top = (1 - u_patch_norm) * depth_tl + u_patch_norm * depth_tr
        depth_bottom = (1 - u_patch_norm) * depth_bl + u_patch_norm * depth_br
        depth_warped = (1 - v_patch_norm) * depth_top + v_patch_norm * depth_bottom
        
        # Mark invalid pixels as far away
        depth_warped = torch.where(valid_patch_pixels, depth_warped, torch.full_like(depth_warped, float('inf')))
        
        # ===== STEP 6: Extract frame and depth crops =====
        frame_crop = frame_t[:3, patch_min_y:patch_max_y, patch_min_x:patch_max_x]  # (3, H_crop, W_crop)
        depth_crop = depth_map_t[patch_min_y:patch_max_y, patch_min_x:patch_max_x]  # (H_crop, W_crop)
        
        # ===== STEP 7: Occlusion handling =====
        valid_depth = (~torch.isnan(depth_crop)) & (depth_crop > 0)
        
        # Depth-adaptive threshold: 3% of mean patch depth, minimum 0.5m.
        # A fixed 5cm threshold causes sigmoid(20*0.05)≈0.73 for on-surface pixels,
        # making the patch appear semi-transparent even when it should be fully opaque.
        # Scaling with depth ensures the sigmoid saturates (~1.0) for on-surface pixels
        # while still correctly occluding when another object is substantially in front.
        mean_patch_depth = torch.mean(patch_corners_3d_t[:, 2])
        occlusion_threshold = torch.clamp(mean_patch_depth * 0.03, min=0.5).item()
        
        # Soft occlusion mask: use sigmoid to create differentiable occlusion weights
        # When depth_warped >> depth_crop: sigmoid → 0 (occluded, patch behind scene)
        # When depth_warped << depth_crop: sigmoid → 1 (visible, patch in front of scene)
        occlusion_temp = 20.0  # Temperature for occlusion sigmoid (tunable: higher = sharper)
        occlusion_weight = torch.sigmoid(occlusion_temp * (depth_crop + occlusion_threshold - depth_warped))  # (H_crop, W_crop)
        
        # Zero out where depth is invalid (no gradient needed for invalid regions)
        occlusion_weight = occlusion_weight * valid_depth.float()
        
        # Combine patch-interior mask with occlusion
        visibility_mask = valid_patch_pixels.float() * occlusion_weight  # (H_crop, W_crop)
        
        # ===== STEP 8: Alpha blending =====
        # Alpha = visibility_mask (patch-interior × occlusion), differentiable w.r.t. depth
        alpha = visibility_mask.unsqueeze(0)  # (1, H_crop, W_crop)
        
        # Alpha blending: result = (1 - alpha) * background + alpha * foreground
        # Differentiable w.r.t. warped_patch (RGB) through grid_sample
        blended_crop = (1 - alpha) * frame_crop + alpha * warped_patch  # (3, H_crop, W_crop)
        
        # Clip to valid range (gradient passes through where not clamped)
        blended_crop = torch.clamp(blended_crop, 0.0, 1.0)
        
        # Assemble full frame: replace the crop region
        # Use in-place-safe construction to maintain gradient flow
        result = frame_t.clone()
        result[:3, patch_min_y:patch_max_y, patch_min_x:patch_max_x] = blended_crop
        
        # Build deployment mask: where the patch is visibly rendered (visibility_mask > 0.5)
        deploy_mask = np.zeros((h, w), dtype=bool)
        deploy_mask[patch_min_y:patch_max_y, patch_min_x:patch_max_x] = \
            (visibility_mask > 0.5).cpu().numpy()
        
        return result, deploy_mask

    def _estimate_corner_depth(self, 
                               points_3d: np.ndarray,
                               u_axis: np.ndarray,
                               v_axis: np.ndarray,
                               centroid: np.ndarray,
                               u_norm: float,
                               v_norm: float) -> float:
        """
        Estimate depth at a specific corner location by sampling nearby surface points.
        
        IMPROVEMENT: Instead of using average depth, compute depth specific to this corner.
        This handles tilted planes correctly.
        
        Args:
            points_3d: All 3D surface points
            u_axis, v_axis: Orthonormal basis vectors on the plane
            centroid: Center of the surface
            u_norm, v_norm: Normalized corner coordinates in [-0.5, 0.5]
            
        Returns:
            Estimated depth (Z-coordinate) at this corner
        """
        # Estimate corner location on plane using basis vectors
        # This projects the normalized corner coordinates onto the actual plane
        # warn: 因为用的是所有的3D点来算max distance，但是3D点可能在平面的内部，这就导致算出来的corner位置：可能会超出surface的区域
        estimated_corner = centroid + \
                            u_norm * 2.0 * np.max(np.abs(points_3d - centroid)) * u_axis + \
                            v_norm * 2.0 * np.max(np.abs(points_3d - centroid)) * v_axis
        
        # Find nearby points in points_3d
        # warn: 没有考虑深度带来的影响啊，这样找到的distance用来近似深度可能会导致非常大的误差？
        distances = np.linalg.norm(points_3d[:, :2] - estimated_corner[:2], axis=1)
        
        # Use median of nearest neighbors to avoid outliers
        n_neighbors = min(10, max(3, len(points_3d) // 4))
        nearest_indices = np.argsort(distances)[:n_neighbors]
        
        # Return median Z of nearby points
        if len(nearest_indices) > 0:
            corner_depth = np.median(points_3d[nearest_indices, 2])
        else:
            # if there is no nearby point, fallback to centroid depth
            corner_depth = centroid[2]
        
        return corner_depth
    
    def _project_point_to_plane(self, 
                               point: np.ndarray,
                               plane_normal: np.ndarray,
                               plane_d: float) -> np.ndarray:
        """
        Project a 3D point onto the fitted plane.
        
        IMPROVEMENT: Ensures all patch corners lie exactly on the plane surface,
        correcting numerical errors and handling non-coplanar drift.
        
        Args:
            point: 3D point to project
            plane_normal: Normal vector of the plane (unit vector)
            plane_d: Plane parameter d (from equation: n·p + d = 0)
            
        Returns:
            Projected point on the plane
        """
        # Plane equation: plane_normal · p + plane_d = 0
        # Distance from point to plane: dist = (plane_normal · point + plane_d)
        
        dist_to_plane = np.dot(plane_normal, point) + plane_d
        
        # Project point onto plane
        projected = point - dist_to_plane * plane_normal
        
        return projected

    def prepare_deployment(self,
                          config: Config,
                          patch_size: Tuple[int, int],
                          surface_mask: np.ndarray) -> DeploymentInfo:
        """Prepare all geometric and tracking data for patch deployment (Steps 1-4).
        
        Computes 3D surface fitting, patch placement, homographies, and optical flow tracking without performing actual rendering. The returned DeploymentInfo can be passed to render_patch_stereo() for rendering.
        
        Args:
            config: Deployment configuration
            patch_size: Tuple of (height, width) for the patch size in pixels
            surface_mask: Binary mask of surface for left view in numpy with shape (h, w) (uint8 {0,1})
            
        Returns:
            DeploymentInfo containing all precomputed data needed for rendering
        """
        
        # ===== STEP 1: Process left view to get 3D surface =====
        print(f"\n[prepare 1/4] Back-projecting surface mask to 3D and fitting plane (frame {config.start_frame_idx})")
        
        start_left_frame = self.dataloader.get_frame(config.start_frame_idx, camera=0)
        
        # Get 3D points from masked region
        # NOTE the following calculation has several issues:
        # 1. it does not consider the occlusion case
        # 2. it does not handle the potential sensor noise
        # 3. not every pixel has the depth ground-truth
        mask_indices = np.where(surface_mask > 0)
        points_3d = []
        
        for y, x in zip(mask_indices[0], mask_indices[1]):
            depth = start_left_frame['depth'][y,x]
            # Check for valid depth (not NaN and positive)
            if not np.isnan(depth) and depth > 0:
                pt_3d = self.back_project(x, y, depth)
                points_3d.append(pt_3d)
        
        points_3d = np.array(points_3d)
        
        if len(points_3d) < 10:
            raise ValueError("Not enough valid depth points in selected region")
        
        # Fit plane which is supposed to be a planar surface
        plane_normal, plane_d = self.fit_plane_ransac(points_3d)
        print(f"  ✓ Fitted plane with {len(points_3d)} 3D points")
        
        # Get UV basis on plane
        u_axis, v_axis = self.compute_uv_basis(points_3d, plane_normal)
        
        # ===== STEP 2: Place patch in 3D (on the plane) =====
        # Strategy: Use centroid depth to compute a single physical size, then lay out a rectangle in the plane's UV space. Project each corner onto the fitted plane so that per-corner depth is derived from plane geometry (exact for planar surfaces).
        print(f"\n[prepare 2/4] Computing patch corner 3D positions on the fitted plane")
        patch_h, patch_w = patch_size
        
        centroid = np.mean(points_3d, axis=0)
        
        # Compute physical patch size using centroid depth (same for all corners → rectangle)
        fx = self.K_left[0, 0]
        fy = self.K_left[1, 1]
        centroid_depth = centroid[2]
        
        # Physical half-extents in meters at centroid depth
        half_width_3d = (patch_w / 2.0 / fx) * centroid_depth
        half_height_3d = (patch_h / 2.0 / fy) * centroid_depth
        
        # Lay out a rectangle centered on the centroid in the plane's UV basis,
        # then project each corner onto the fitted plane for exact coplanarity.
        # Corner order: TL, TR, BR, BL (matching patch pixel corners)
        corner_uv_offsets = np.array([
            [-half_width_3d, -half_height_3d],  # TL
            [ half_width_3d, -half_height_3d],  # TR
            [ half_width_3d,  half_height_3d],  # BR
            [-half_width_3d,  half_height_3d],  # BL
        ])
        
        patch_corners_3d_left = []
        for uv_offset in corner_uv_offsets:
            pt_3d = centroid + uv_offset[0] * u_axis + uv_offset[1] * v_axis
            # Project onto the fitted plane to ensure exact coplanarity and correct depth
            pt_3d = self._project_point_to_plane(pt_3d, plane_normal, plane_d)
            patch_corners_3d_left.append(pt_3d)
        
        patch_corners_3d_left = np.array(patch_corners_3d_left)
        
        # Convert to right camera coordinates
        patch_corners_3d_right = []
        for pt_3d in patch_corners_3d_left:
            pt_3d_right = self.R @ pt_3d + self.T.flatten()
            patch_corners_3d_right.append(pt_3d_right)
        
        patch_corners_3d_right = np.array(patch_corners_3d_right)
        
        # ===== STEP 3: Compute initial homographies =====
        print(f"\n[prepare 3/4] Computing patch-to-frame homographies for left and right views")
        
        src_corners = np.array([
            [0, 0], [patch_w, 0], [patch_w, patch_h], [0, patch_h]
        ], dtype=float)
        
        # Left view homography
        dst_corners_left = []
        for pt_3d in patch_corners_3d_left:
            u, v = self.project_3d_to_left(pt_3d)
            dst_corners_left.append([u, v])
        dst_corners_left = np.array(dst_corners_left)
        
        H_patch_to_left = cv2.getPerspectiveTransform(
            src_corners.astype(np.float32),
            dst_corners_left.astype(np.float32)
        )
        print(f"  ✓ Left camera perspective transform computed")
        
        # Right view homography
        # NOTE: project_3d_to_right expects LEFT camera coordinates
        # (it applies the left→right extrinsic transform internally)
        dst_corners_right = []
        for pt_3d in patch_corners_3d_left:
            u, v = self.project_3d_to_right(pt_3d)
            dst_corners_right.append([u, v])
        dst_corners_right = np.array(dst_corners_right)
        
        H_patch_to_right = cv2.getPerspectiveTransform(
            src_corners.astype(np.float32),
            dst_corners_right.astype(np.float32)
        )
        print(f"  ✓ Right camera perspective transform computed")
        
        # ===== STEP 4: Track surface using optical flow (left view) =====
        print(f"\n[prepare 4/4] Tracking surface across frames with optical flow and depth mapping")
        
        print("\n\t[a/] Accumulating frame-to-frame homographies of left view via flow-based point tracking")
        forward_homographies = self.track_surface_with_flow(
            config.start_frame_idx, surface_mask, 'forward'
        )
        backward_homographies = self.track_surface_with_flow(
            config.start_frame_idx, surface_mask, 'backward'
        )

        all_homographies_left = backward_homographies[::-1][:-1] + forward_homographies

        print("\n\t[b/] Accumulating frame-to-frame homographies of right view via flow-based point tracking")
        # Compute right-view surface mask by warping left mask using planar homography
        H_left_to_right = cv2.getPerspectiveTransform(
            dst_corners_left.astype(np.float32),
            dst_corners_right.astype(np.float32)
        )
        h_img, w_img = surface_mask.shape[:2]
        surface_mask_right = cv2.warpPerspective(
            surface_mask, H_left_to_right, (w_img, h_img),
            flags=cv2.INTER_NEAREST
        )

        forward_homographies_right = self.track_surface_with_flow(
            config.start_frame_idx, surface_mask_right, 'forward', camera=1
        )
        backward_homographies_right = self.track_surface_with_flow(
            config.start_frame_idx, surface_mask_right, 'backward', camera=1
        )

        all_homographies_right = backward_homographies_right[::-1][:-1] + forward_homographies_right
        
        print("\n\t[c/] Warping surface mask with flow and checking per-frame visibility")
        
        forward_masks, forward_visibility = self.track_surface_mask_with_flow(
            config.start_frame_idx, surface_mask, 'forward'
        )
        backward_masks, backward_visibility = self.track_surface_mask_with_flow(
            config.start_frame_idx, surface_mask, 'backward'
        )

        all_masks = backward_masks[::-1][:-1] + forward_masks
        all_visibility = backward_visibility[::-1][:-1] + forward_visibility

        assert len(all_masks) == self.n_frames and len(all_visibility) == self.n_frames and len(all_homographies_left) == self.n_frames and len(all_homographies_right) == self.n_frames, "Tracking results length mismatch!"
        
        # ===== summarize deployment statistics =====
        visible_indices = [i for i, v in enumerate(all_visibility) if v]
        if visible_indices:
            start_idx = visible_indices[0]
            end_idx = visible_indices[-1]
        else:
            raise ValueError("Surface is not visible in any frame after tracking!")

        visible_frame_count = len(visible_indices)
        not_visible_frame_count = self.n_frames - visible_frame_count

        print(f"\t  Deployment frame range: [{start_idx}, {end_idx}]")
        print(f"\t  ✓ Visible frames: {visible_frame_count}")
        print(f"\t  ✗ Not visible frames: {not_visible_frame_count}")
        
        return DeploymentInfo(
            H_patch_to_left=H_patch_to_left,
            H_patch_to_right=H_patch_to_right,
            all_homographies_left=all_homographies_left,
            all_homographies_right=all_homographies_right,
            all_masks=all_masks,
            all_visibility=all_visibility,
            patch_corners_3d_left=patch_corners_3d_left,
            patch_corners_3d_right=patch_corners_3d_right,
            ref_frame_idx=config.start_frame_idx,
            start_idx=start_idx,
            end_idx=end_idx,
        )

    def render_patch_stereo(self,
                           deployment_info: DeploymentInfo,
                           patch_img: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[np.ndarray], range]:
        """Render given patch into all stereo frames using precomputed deployment info.
        
        This is the actual rendering step, separated from prepare_deployment() so that the same deployment geometry can be reused with different patch images (e.g., during gradient-based optimization).
        
        Args:
            deployment_info: Precomputed deployment data from prepare_deployment() function
            patch_img: RGB patch image as tensor with shape (3, h, w) (float32 [0, 1])
            
        Returns:
            results_left: list of torch.Tensor (3, H, W) float32 [0, 1] RGB frames
            results_right: list of torch.Tensor (3, H, W) float32 [0, 1] RGB frames
            deploy_masks: per-frame boolean masks (H, W) showing where the patch is rendered (left view)
            visible_range: range of visible frame indices
        """
        info = deployment_info
        
        # ===== Render patch in all frames for both views =====
        print(f"\n[render] Rendering patch into left and right views for {self.n_frames} frames")
        
        results_left = [None] * self.n_frames
        results_right = [None] * self.n_frames
        deploy_masks = [None] * self.n_frames
        
        frames_with_patch = 0
        frames_not_visible = 0
        frames_missing_depth = 0
        
        # Get reference frame extrinsic (where 3D corners were computed)
        T_ref_left = self.dataloader.get_frame_extrinsic(info.ref_frame_idx, camera_id=0)
        T_ref_right = self.dataloader.get_frame_extrinsic(info.ref_frame_idx, camera_id=1)
        
        # Homogeneous corner coordinates: (4, 4) with rows [x, y, z, 1]
        ones_col = np.ones((4, 1), dtype=np.float64)
        corners_left_homo = np.hstack([info.patch_corners_3d_left, ones_col])   # (4, 4)
        corners_right_homo = np.hstack([info.patch_corners_3d_right, ones_col])  # (4, 4)
        
        for i in range(self.n_frames):
            stereo_frame = self.dataloader.get_stereo_pair(frame_idx=i)
            
            # permute stereo frames to facilitate downstream processing
            left_frame = (stereo_frame['left']['rgb'].astype(np.float32) / 255.0).transpose(2,0,1)  # (C, H, W)
            right_frame = (stereo_frame['right']['rgb'].astype(np.float32) / 255.0).transpose(2,0,1)
            
            # Check if mask is visible in this frame
            if not info.all_visibility[i]:
                frames_not_visible += 1
                results_left[i] = left_frame.copy()
                results_right[i] = right_frame.copy()
                h_frame, w_frame = stereo_frame['left']['rgb'].shape[:2]
                deploy_masks[i] = np.zeros((h_frame, w_frame), dtype=bool)
                continue

            # Check if depth maps are available
            if stereo_frame['left']['depth'] is None or stereo_frame['right']['depth'] is None:
                frames_missing_depth += 1
                results_left[i] = left_frame.copy()
                results_right[i] = right_frame.copy()
                h_frame, w_frame = stereo_frame['left']['rgb'].shape[:2]
                deploy_masks[i] = np.zeros((h_frame, w_frame), dtype=bool)
                continue
            
            # Transform patch corners from ref frame's camera coords to frame i's camera coords
            # T_ref_to_frame_i = T_world_to_cam_i @ inv(T_world_to_cam_ref)
            T_i_left = self.dataloader.get_frame_extrinsic(i, camera_id=0)
            T_i_right = self.dataloader.get_frame_extrinsic(i, camera_id=1)
            
            if T_i_left is not None and T_ref_left is not None:
                T_ref_to_i_left = T_i_left @ np.linalg.inv(T_ref_left)
                corners_3d_left_i = (T_ref_to_i_left @ corners_left_homo.T).T[:, :3]
            else:
                corners_3d_left_i = info.patch_corners_3d_left  # fallback
            
            if T_i_right is not None and T_ref_right is not None:
                T_ref_to_i_right = T_i_right @ np.linalg.inv(T_ref_right)
                corners_3d_right_i = (T_ref_to_i_right @ corners_right_homo.T).T[:, :3]
            else:
                corners_3d_right_i = info.patch_corners_3d_right  # fallback
            
            # Left view
            H_patch_to_frame_left = info.all_homographies_left[i] @ info.H_patch_to_left
            results_left[i], deploy_masks[i] = self.render_patch_torch(
                stereo_frame['left']['rgb'],
                stereo_frame['left']['depth'],
                patch_img,
                H_patch_to_frame_left,
                corners_3d_left_i
            )
            
            # Right view (mask from left view is used for loss computation)
            H_patch_to_frame_right = info.all_homographies_right[i] @ info.H_patch_to_right
            results_right[i], _ = self.render_patch_torch(
                stereo_frame['right']['rgb'],
                stereo_frame['right']['depth'],
                patch_img,
                H_patch_to_frame_right,
                corners_3d_right_i
            )
            
            frames_with_patch += 1
        
        print(f"\n[render] Summary:")
        print(f"  ✓ Patch rendered in: [{info.start_idx}-{info.end_idx}] {frames_with_patch}/{self.n_frames} frames")
        print(f"  • Surface not visible: {frames_not_visible} frames")
        print(f"  • Missing depth maps: {frames_missing_depth} frames")
        
        return results_left, results_right, deploy_masks, range(info.start_idx, info.end_idx+1)

    def deploy_patch_stereo(self,
                           config: Config,
                           patch_img: torch.Tensor,
                           surface_mask: np.ndarray) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[np.ndarray], range]:
        """Deploy patch to both stereo views using 3D placement.
        
        Convenience method that calls prepare_deployment() followed by render_patch_stereo().
        For iterative optimization, call those two methods separately so geometry is computed once.
        
        Args:
            config: Deployment configuration
            patch_img: RGB patch image as tensor with shape (3, h, w) (float32 [0, 1])
            surface_mask: Binary mask of surface for left view in numpy with shape (h, w) (uint8 {0,1})
            
        Returns:
            results_left: list of torch.Tensor (3, H, W) float32 [0, 1] RGB frames
            results_right: list of torch.Tensor (3, H, W) float32 [0, 1] RGB frames
            deploy_masks: per-frame boolean masks (H, W) showing where the patch is rendered (left view)
            visible_range: range of visible frame indices
        """
        deployment_info = self.prepare_deployment(config, patch_img, surface_mask)
        return self.render_patch_stereo(deployment_info, patch_img)





# ===== Helper Functions =====
def save_debug_image(image: np.ndarray, filename: str, convert_rgb_to_bgr: bool = False) -> None:
    """
    Save an image using cv2.imwrite.
    
    Args:
        image: Image array to save (numpy.ndarray)
        filename: Output filename/path
        convert_rgb_to_bgr: If True, convert RGB to BGR before saving (since cv2 uses BGR)
    
    Returns:
        None
    """
    if convert_rgb_to_bgr and image.shape[2] == 3:
        # Convert RGB to BGR for cv2
        image_to_save = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image_to_save = image
    
    # Ensure image is in the right format (uint8)
    if image_to_save.dtype != np.uint8:
        if image_to_save.max() <= 1.0:
            # Normalize from [0, 1] to [0, 255]
            image_to_save = (image_to_save * 255).astype(np.uint8)
        else:
            # Clip and convert
            image_to_save = np.clip(image_to_save, 0, 255).astype(np.uint8)
    
    cv2.imwrite(filename, image_to_save)
    print(f"Saved debug image: {filename}")

def check_stereo_consistency(results_left, results_right, frame_idx):
    """Check if patch appears at correct depth in stereo."""
    left_frame = results_left[frame_idx]
    right_frame = results_right[frame_idx]
    
    # Simple check: patch should be displaced horizontally
    # according to its depth (disparity = baseline * focal_length / depth)
    
    # Convert to grayscale for correlation
    left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
    
    # Compute stereo disparity (simplified)
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=64,
        blockSize=11
    )
    disparity = stereo.compute(left_gray, right_gray)
    
    # Check if patch regions have plausible disparity
    return disparity

def validate_patch_size(surface_mask: np.ndarray, patch_img: np.ndarray, 
                        cfg) -> np.ndarray:
    """
    Validate and adjust patch size based on surface mask and configuration.
    
    Maintains the user-specified aspect ratio while respecting available surface space.
    
    Args:
        surface_mask: Binary mask of deployable surface region
        patch_img: Original patch image
        cfg: Configuration object with patch.mode and size parameters
        
    Returns:
        Resized patch image with valid dimensions
    """
    (x, y, max_width, max_height), area = find_maximum_rectangle(surface_mask)
    print(f"Maximum rectangle: position=({x}, {y}), size=({max_width}x{max_height}), area={area}")
    
    if cfg.patch.mode == 'given_size':
        assert cfg.patch.given_width is not None and cfg.patch.given_height is not None, \
            "Given width and height must be specified when deploy_mode is 'given_size'"
        
        if cfg.patch.given_width <= max_width and cfg.patch.given_height <= max_height:
            width, height = cfg.patch.given_width, cfg.patch.given_height
        else:
            # Scale down while maintaining aspect ratio
            scale_w = max_width / cfg.patch.given_width
            scale_h = max_height / cfg.patch.given_height
            scale = min(scale_w, scale_h)
            
            width = int(cfg.patch.given_width * scale)
            height = int(cfg.patch.given_height * scale)
            
            print(f"Given size ({cfg.patch.given_width}x{cfg.patch.given_height}) exceeds "
                  f"maximum available space ({max_width}x{max_height}). "
                  f"Scaled to ({width}x{height}) maintaining aspect ratio.")
    
    elif cfg.patch.mode == 'maximum_size':
        # Maximize size while maintaining the user-specified aspect ratio
        scale_w = max_width / cfg.patch.given_width
        scale_h = max_height / cfg.patch.given_height
        scale = min(scale_w, scale_h)
        
        width = int(cfg.patch.given_width * scale)
        height = int(cfg.patch.given_height * scale)
        
        print(f"Maximizing patch size with aspect ratio {target_aspect_ratio:.4f}: "
              f"({width}x{height}) in available space ({max_width}x{max_height})")
    
    else:
        raise ValueError(f"Invalid deploy mode: {cfg.patch.mode}")
    
    return cv2.resize(patch_img, (width, height), interpolation=cv2.INTER_AREA)

# ===== Example Usage =====
if __name__ == "__main__":
    
    from dataset import VirtualKITTI2Loader

    # Load configuration
    cfg = Config.fromfile('/home/yxing/projects/stereo_PhysicalAttack/src/config/temporal/dpa_temp.py')

    # Initialize scene loader
    loader = VirtualKITTI2Loader(
        root_dir=cfg.dataset.root,
        scene=cfg.dataset.scene,
        variation=cfg.dataset.variation
    )

    # Initialize patch deployer
    deployer = StereoPatchDeployer(sceneloader=loader)
    
    # Load patch
    patch_img = cv2.imread(cfg.patch.file, cv2.IMREAD_UNCHANGED) # load to including alpha channel if exists, ndarray, uint8, (h, w, c), BGR
    
    # Convert BGR to RGB to match internal frame format
    if patch_img.shape[2] == 4:
        # BGRA -> RGBA
        patch_img = cv2.cvtColor(patch_img, cv2.COLOR_BGRA2RGBA)
    else:
        # BGR -> RGB, then add alpha channel
        patch_img = cv2.cvtColor(patch_img, cv2.COLOR_BGR2RGB)
        alpha = np.ones((patch_img.shape[0], patch_img.shape[1], 1), dtype=patch_img.dtype) * 255
        patch_img = np.concatenate([patch_img, alpha], axis=2)
    
    # User selection: initial frame, deploy region
    surface_mask_left = cv2.imread(cfg.deploy.frame_mask_left, cv2.IMREAD_UNCHANGED) # load to including alpha channel if exists, ndarray, uint8, (h, w, c), BGR

    # Validate and adjust patch size
    # patch_img = validate_patch_size(surface_mask_left, patch_img, cfg)
    
    # Deploy patch
    results_left, results_right = deployer.deploy_patch_stereo(
        config=cfg,
        patch_img=patch_img,
        surface_mask=surface_mask_left,
    )
    
    # Save results
    height, width = results_left[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    out_left = cv2.VideoWriter('output_left.mp4', fourcc, 10.0, (width, height))
    out_right = cv2.VideoWriter('output_right.mp4', fourcc, 10.0, (width, height))
    
    for frame_left, frame_right in zip(results_left, results_right):
        # Convert from RGB (internal format) to BGR (cv2.VideoWriter format)
        frame_left_bgr = cv2.cvtColor(frame_left, cv2.COLOR_RGB2BGR)
        frame_right_bgr = cv2.cvtColor(frame_right, cv2.COLOR_RGB2BGR)
        out_left.write(frame_left_bgr)
        out_right.write(frame_right_bgr)
    
    out_left.release()
    out_right.release()
    
    print("Saved: output_left.mp4, output_right.mp4")
    
    # Save stereo pair for verification
    stereo_pair = results_left[235] # np.hstack([results_left[235], results_right[235]])
    # Convert from RGB to BGR for cv2.imwrite
    stereo_pair_bgr = cv2.cvtColor(stereo_pair, cv2.COLOR_RGB2BGR)
    cv2.imwrite('stereo_pair.png', stereo_pair_bgr)
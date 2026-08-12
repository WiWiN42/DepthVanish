'''
Author: Zachery Berger <zackeberger@g.ucla.edu>, Parth Agrawal <parthagrawal24@g.ucla.edu>, Tian Yu Liu <tianyu139@g.ucla.edu>, Alex Wong <alexw@cs.ucla.edu>
If you use this code, please cite the following paper:

Z. Berger, P. Agrawal, T. Liu, S. Soatto, and A. Wong. Stereoscopic Universal Perturbations across Different Architectures and Datasets.
https://arxiv.org/pdf/2112.06116.pdf

@inproceedings{berger2022stereoscopic,
  title={Stereoscopic Universal Perturbations across Different Architectures and Datasets},
  author={Berger, Zachery and Agrawal, Parth and Liu, Tian Yu and Soatto, Stefano and Wong, Alex},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
'''

import re
import cv2
from pathlib import Path
import numpy as np
import torch.utils.data
from typing import Dict, List, Tuple, Optional, Union

from utils.tool import read_paths, load_image, load_disparity

class StereoDataset(torch.utils.data.Dataset):
    '''
    Loads a stereo pair, and if available ground truth and pseudo ground truth

    Arg(s):
        image0_paths : list[str]
            path to left image
        image1_paths : list[str]
            path to right image
        ground_truth_paths : list[str]
            path to disparity ground truth
        pseudo_ground_truth_paths : list[str]
            path to pseudo (from another model) ground truth disparity
            (H, W) to resize images
    '''

    def __init__(self,
                 image0_paths,
                 image1_paths,
                 ground_truth_paths=None,
                 pseudo_ground_truth_paths=None,
                 norm=None):

        self.norm = norm
        self.image0_paths = read_paths(image0_paths)
        self.image1_paths = read_paths(image1_paths)

        assert len(self.image0_paths) == len(self.image1_paths)

        if ground_truth_paths is None:
            self.ground_truth_paths = [None] * len(self.image0_paths)
        else:
            self.ground_truth_paths = ground_truth_paths

        if pseudo_ground_truth_paths is None:
            self.pseudo_ground_truth_paths = [None] * len(self.image0_paths)
        else:
            self.pseudo_ground_truth_paths = pseudo_ground_truth_paths

    def __getitem__(self, index):
        # Load images
        image0 = load_image(
            self.image0_paths[index])

        image1 = load_image(
            self.image1_paths[index])

        # Load ground truth, if not available then return zeros
        ground_truth_path = self.ground_truth_paths[index]

        if ground_truth_path is not None:
            ground_truth = load_disparity(ground_truth_path)
        else:
            ground_truth = torch.zeros([1] + list(image0.shape[1:3]))

        pseudo_ground_truth_path = self.pseudo_ground_truth_paths[index]

        if pseudo_ground_truth_path is not None:
            pseudo_ground_truth = load_disparity(pseudo_ground_truth_path)
        else:
            pseudo_ground_truth = torch.zeros([1] + list(image0.shape[1:3]))

        if self.norm is not None:
            image0, image1 = self.normalize(
                [image0, image1], normalized_image_range=self.norm)

        return image0, image1, ground_truth, pseudo_ground_truth

    @classmethod
    def normalize(cls, images_tensor, normalized_image_range=[0, 1]):
        '''
        Normalize image to a given range

        Arg(s):
            images_tensor : list[torch.Tensor[float32]]
                list of N x C x H x W tensors
            normalized_image_range : list[float]
                intensity range after normalizing images
        Returns:
            list[torch.Tensor[float32]] : list of normalized N x C x H x W tensors
        '''

        if normalized_image_range == [0, 1]:
            images_tensor = [
                images / 255.0 for images in images_tensor
            ]
        elif normalized_image_range == [-1, 1]:
            images_tensor = [
                2.0 * (images / 255.0) - 1.0 for images in images_tensor
            ]
        elif normalized_image_range == [0, 255]:
            pass
        else:
            raise ValueError('Unsupported normalization range: {}'.format(
                normalized_image_range))

        return images_tensor

    def __len__(self):
        return len(self.image0_paths)

class VirtualKITTI2Loader:
    """
    Loader for Virtual KITTI 2 dataset.
    Handles frames, depth, optical flow, and camera parameters.
    """

    def __init__(self, root_dir: Union[str, Path], scene: str, variation: str = "clone"):
        """
        Initialize the loader for a specific scene and variation of Virtual KITTI 2 dataset.
        
        Args:
            root_dir: Path to the Virtual KITTI 2 dataset root directory
            scene: Scene name (e.g., "0001", "0002", etc.)
            variation: Variation type - one of: 
                      "clone", "fog", "morning", "overcast", "rain", "sunset"
        """
        SCENE_ID = ['01', '02', '06', '18', '20']

        VARIATIONS = [
        '15-deg-left', 'clone', 'overcast', 'sunset', '15-deg-right', '30-deg-left', 'rain', '30-deg-right', 'fog', 'morning'
        ]
        self.root_dir = Path(root_dir)
        assert scene in SCENE_ID, f"Invalid scene: {scene}"
        self.scene = scene
        assert variation in VARIATIONS, f"Invalid scene variation: {variation}"
        self.variation = variation
        
        # Validate paths
        self.scene_dir = self.root_dir / f'Scene{scene}' / variation
        if not self.scene_dir.exists():
            raise ValueError(f"Scene directory not found: {self.scene_dir}")
        
        # File pattern templates
        self.file_templates = {
            "rgb": "frames/rgb/Camera_{}/rgb_{}.jpg",
            "depth": "frames/depth/Camera_{}/depth_{}.png",
            "forward_flow": "frames/forwardFlow/Camera_{}/flow_{}.png",
            "backward_flow": "frames/backwardFlow/Camera_{}/backwardFlow_{}.png",
            # "segmentation": "frames/segmentation/Camera_{}/segmentation_{:05d}.png",
            # "instance": "frames/instanceSegmentation/Camera_{}/instanceSegmentation_{:05d}.png"
        }
        
        # Load camera parameters
        self.camera_params = self._load_camera_parameters()
        
        # Get available files
        self.file_indices = self._get_file_indices()
        
        # Create mapping from file index to frame index (they may differ)
        self.frame_indices = self._create_frame_mapping()
    
    def _get_file_indices(self) -> List[str]:
        """Extract all available file indices from the RGB directory."""
        rgb_dir = self.scene_dir / "frames" / "rgb" / "Camera_0"
        if not rgb_dir.exists():
            # Try another camera
            rgb_dir = self.scene_dir / "frames" / "rgb" / "Camera_1"
            if not rgb_dir.exists():
                # Try alternative structure
                rgb_dir = self.scene_dir / "rgb" / "Camera_0"
        
        file_indices = []
        pattern = re.compile(r'rgb_(\d{5})\.jpg')
        
        for file in rgb_dir.glob("*.jpg"):
            match = pattern.match(file.name)
            if match:
                file_indices.append(match.group(1))
        
        return sorted(file_indices)
    
    def _create_frame_mapping(self) -> Dict[int, str]:
        """Create mapping from dataset frame index to file naming index."""
        # In case of file indices start from 00001, but extrinsic frame indices start from 0
        if self.file_indices:
            # min_idx = min(self.file_indices)
            mapping = {i: idx for i, idx in enumerate(self.file_indices)}
            return mapping
        return {}
    
    def _load_camera_parameters(self) -> Dict:
        """
        Load camera intrinsic and extrinsic parameters.
        
        Returns:
            Dictionary with complete camera parameters
        """
        params = {
            "Camera_0": {}, 
            "Camera_1": {},
            "stereo": {}
        }
        
        # ============================
        # 1. INTRINSIC PARAMETERS
        # ============================
        # Virtual KITTI 2 uses fixed intrinsic parameters
        K = np.array([
            [725.0087, 0.0,      620.5],
            [0.0,      725.0087, 187.0],
            [0.0,      0.0,      1.0]
        ], dtype=np.float32)
        
        params["Camera_0"]["intrinsic"] = K
        params["Camera_1"]["intrinsic"] = K.copy()
        
        # Image dimensions
        params["image_width"] = 1242
        params["image_height"] = 375
        
        # ============================
        # 2. EXTRINSIC PARAMETERS
        # ============================
        # Camera parameters file
        camera_params_file = self.scene_dir / "extrinsic.txt"
        if camera_params_file.exists():
            # Parse extrinsic parameters
            extrinsic_data = self._parse_extrinsic_matrix(camera_params_file)
            
            if extrinsic_data:
                # Store all extrinsic data for frame-by-frame access
                params["extrinsic_data"] = extrinsic_data
                
                # Get transformation for Camera 0 (left) and Camera 1 (right)
                # Use frame 0 as reference for computing stereo parameters
                cam0_data = extrinsic_data.get(0, {})
                cam1_data = extrinsic_data.get(1, {})
                
                if cam0_data and cam1_data:
                    # Get first available frame for each camera
                    frame_0 = sorted(cam0_data.keys())[0]
                    T_world_to_cam0_ref = cam0_data[frame_0]
                    
                    frame_1 = sorted(cam1_data.keys())[0]
                    T_world_to_cam1_ref = cam1_data[frame_1]
                    
                    # Store reference transformations
                    params["Camera_0"]["T_world_to_cam_ref"] = T_world_to_cam0_ref
                    params["Camera_1"]["T_world_to_cam_ref"] = T_world_to_cam1_ref
                    
                    # Extract rotation and translation from reference
                    R0_ref = T_world_to_cam0_ref[:3, :3]
                    t0_ref = T_world_to_cam0_ref[:3, 3]
                    
                    R1_ref = T_world_to_cam1_ref[:3, :3]
                    t1_ref = T_world_to_cam1_ref[:3, 3]
                    
                    params["Camera_0"]["rotation_ref"] = R0_ref
                    params["Camera_0"]["translation_ref"] = t0_ref
                    params["Camera_1"]["rotation_ref"] = R1_ref
                    params["Camera_1"]["translation_ref"] = t1_ref
                    
                    # ============================
                    # 3. RELATIVE TRANSFORMATION (Camera 0 to Camera 1)
                    # ============================
                    T_cam0_to_world = np.linalg.inv(T_world_to_cam0_ref)
                    T_cam0_to_cam1 = T_world_to_cam1_ref @ T_cam0_to_world
                    
                    params["stereo"]["T_left_to_right"] = T_cam0_to_cam1
                    
                    # Extract rotation and translation from relative transformation
                    R_left_to_right = T_cam0_to_cam1[:3, :3]
                    t_left_to_right = T_cam0_to_cam1[:3, 3]
                    
                    params["stereo"]["R_left_to_right"] = R_left_to_right
                    params["stereo"]["t_left_to_right"] = t_left_to_right
                    
                    # Baseline
                    baseline = np.linalg.norm(t_left_to_right)
                    params["stereo"]["baseline"] = baseline
                    
                    # ============================
                    # 4. PROJECTION MATRICES (reference)
                    # ============================
                    P0_ref = K @ T_world_to_cam0_ref[:3, :]
                    P1_ref = K @ T_world_to_cam1_ref[:3, :]

                    params["Camera_0"]["projection_ref"] = P0_ref
                    params["Camera_1"]["projection_ref"] = P1_ref
                    
                    # Rectified projection matrices
                    P0_rect = np.zeros((3, 4), dtype=np.float32)
                    P1_rect = np.zeros((3, 4), dtype=np.float32)
                    
                    P0_rect[:3, :3] = K
                    P1_rect[:3, :3] = K
                    P1_rect[0, 3] = -K[0, 0] * baseline
                    
                    params["Camera_0"]["projection_rect"] = P0_rect
                    params["Camera_1"]["projection_rect"] = P1_rect
                    
                    # Pose matrices (reference)
                    params["Camera_0"]["T_cam_to_world_ref"] = np.linalg.inv(T_world_to_cam0_ref)
                    params["Camera_1"]["T_cam_to_world_ref"] = np.linalg.inv(T_world_to_cam1_ref)
                else:
                    print(f"Warning: Could not find both camera 0 and 1 in extrinsic data")
                    params = self._load_default_camera_params(params)
            else:
                print(f"Warning: Could not parse extrinsic file")
                params = self._load_default_camera_params(params)
        else:
            print(f"Warning: Camera parameter file not found at {camera_params_file}")
            params = self._load_default_camera_params(params)
        
        return params
    
    def _parse_extrinsic_matrix(self, extrinsic_file: str) -> Dict[int, Dict[int, np.ndarray]]:
        """
        Parse extrinsic parameters from file.
        
        Format: frame cameraID r1,1 r1,2 r1,3 t1 r2,1 r2,2 r2,3 t2 r3,1 r3,2 r3,3 t3 0 0 0 1
        
        Returns:
            Dictionary: cameraID -> frame -> 4x4 transformation matrix
        """
        extrinsic_data = {}
        
        try:
            with open(extrinsic_file, 'r') as f:
                lines = f.readlines()
                
                # the first line is supposed to be header 
                start_idx = 1
                
                for line in lines[start_idx:]:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Split line
                    parts = re.split(r'\s+', line.strip())
                    
                    if len(parts) != 18:  # frame + cameraID + 16 matrix elements
                        print(f"Warning: Skipping line with {len(parts)} parts: {line[:50]}...") # just warning
                        continue
                    
                    try:
                        # Parse frame and camera ID
                        frame_idx = int(parts[0])
                        camera_id = int(parts[1])
                        
                        # Parse the 4x4 transformation matrix elements
                        matrix_elements = list(map(float, parts[2:]))
                        
                        # Reshape to 4x4 matrix
                        T = np.array(matrix_elements, dtype=np.float32).reshape(4, 4)
                        
                        # Ensure last row is [0, 0, 0, 1]
                        if not np.allclose(T[3, :], [0, 0, 0, 1]):
                            T[3, :] = [0, 0, 0, 1]
                        
                        # Store in dictionary
                        if camera_id not in extrinsic_data:
                            extrinsic_data[camera_id] = {}
                        
                        extrinsic_data[camera_id][frame_idx] = T                       
                    except (ValueError, IndexError) as e:
                        print(f"Warning: Could not parse line: {line[:50]}... Error: {e}")
                        continue
                        
        except Exception as e:
            print(f"Error parsing extrinsic file: {e}")
        
        return extrinsic_data
    
    def _load_default_camera_params(self, params: Dict) -> Dict:
        """Load default camera parameters when extrinsic file is not available."""
        baseline = 0.532725  # meters
        K = params["Camera_0"]["intrinsic"]
        
        # Left camera (identity)
        params["Camera_0"]["rotation_ref"] = np.eye(3, dtype=np.float32)
        params["Camera_0"]["translation_ref"] = np.zeros(3, dtype=np.float32)
        params["Camera_0"]["T_world_to_cam_ref"] = np.eye(4, dtype=np.float32)
        params["Camera_0"]["projection_ref"] = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        params["Camera_0"]["T_cam_to_world_ref"] = np.eye(4, dtype=np.float32)
        
        # Right camera (translated along X)
        T_right = np.array([
            [1, 0, 0, baseline],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        params["Camera_1"]["rotation_ref"] = np.eye(3, dtype=np.float32)
        params["Camera_1"]["translation_ref"] = np.array([baseline, 0, 0], dtype=np.float32)
        params["Camera_1"]["T_world_to_cam_ref"] = T_right
        params["Camera_1"]["projection_ref"] = K @ np.hstack([np.eye(3), np.array([[baseline, 0, 0]]).T])
        params["Camera_1"]["T_cam_to_world_ref"] = np.linalg.inv(T_right)
        
        # Stereo parameters
        params["stereo"]["R_left_to_right"] = np.eye(3, dtype=np.float32)
        params["stereo"]["t_left_to_right"] = np.array([baseline, 0, 0], dtype=np.float32)
        params["stereo"]["T_left_to_right"] = T_right
        params["stereo"]["baseline"] = baseline
        
        # Rectified projection matrices
        P0_rect = np.zeros((3, 4), dtype=np.float32)
        P1_rect = np.zeros((3, 4), dtype=np.float32)
        P0_rect[:3, :3] = K
        P1_rect[:3, :3] = K
        P1_rect[0, 3] = -K[0, 0] * baseline
        
        params["Camera_0"]["projection_rect"] = P0_rect
        params["Camera_1"]["projection_rect"] = P1_rect
        
        return params
    
    def get_frame_extrinsic(self, frame_idx: int, camera_id: int = 0) -> Optional[np.ndarray]:
        """Get extrinsic matrix for a specific frame and camera."""
        if "extrinsic_data" in self.camera_params:
            cam_data = self.camera_params["extrinsic_data"].get(camera_id, {})
            return cam_data.get(frame_idx, None)
        return None
    
    def get_camera_parameters_summary(self) -> Dict:
        """Get a summary of all camera parameters."""
        summary = {
            "intrinsics": {
                "left": self.camera_params["Camera_0"]["intrinsic"].tolist(),
                "right": self.camera_params["Camera_1"]["intrinsic"].tolist(),
                "image_size": [self.camera_params["image_height"], 
                              self.camera_params["image_width"]]
            },
            "extrinsics_ref": {
                "left": {
                    "rotation": self.camera_params["Camera_0"]["rotation_ref"].tolist(),
                    "translation": self.camera_params["Camera_0"]["translation_ref"].tolist(),
                    "T_world_to_cam": self.camera_params["Camera_0"]["T_world_to_cam_ref"].tolist()
                },
                "right": {
                    "rotation": self.camera_params["Camera_1"]["rotation_ref"].tolist(),
                    "translation": self.camera_params["Camera_1"]["translation_ref"].tolist(),
                    "T_world_to_cam": self.camera_params["Camera_1"]["T_world_to_cam_ref"].tolist()
                }
            },
            "stereo": {
                "R_left_to_right": self.camera_params["stereo"]["R_left_to_right"].tolist(),
                "t_left_to_right": self.camera_params["stereo"]["t_left_to_right"].tolist(),
                "T_left_to_right": self.camera_params["stereo"]["T_left_to_right"].tolist(),
                "baseline": float(self.camera_params["stereo"]["baseline"])
            },
            "projections": {
                "left_ref": self.camera_params["Camera_0"]["projection_ref"].tolist(),
                "right_ref": self.camera_params["Camera_1"]["projection_ref"].tolist(),
                "left_rectified": self.camera_params["Camera_0"]["projection_rect"].tolist(),
                "right_rectified": self.camera_params["Camera_1"]["projection_rect"].tolist()
            }
        }
        
        return summary
    
    def compute_fundamental_matrix(self, frame_idx: int = 0) -> np.ndarray:
        """
        Compute fundamental matrix for a specific frame.
        """
        # Get extrinsics for this frame
        T_cam0 = self.get_frame_extrinsic(frame_idx, 0)
        T_cam1 = self.get_frame_extrinsic(frame_idx, 1)
        
        if T_cam0 is None or T_cam1 is None:
            # Use reference if frame-specific not available
            T_cam0 = self.camera_params["Camera_0"]["T_world_to_cam_ref"]
            T_cam1 = self.camera_params["Camera_1"]["T_world_to_cam_ref"]
        
        K = self.camera_params["Camera_0"]["intrinsic"]
        
        # Compute relative transformation
        T_cam0_to_world = np.linalg.inv(T_cam0)
        T_cam0_to_cam1 = T_cam1 @ T_cam0_to_world
        
        R = T_cam0_to_cam1[:3, :3]
        t = T_cam0_to_cam1[:3, 3]
        
        # Skew-symmetric matrix for translation
        t_x = np.array([
            [0, -t[2], t[1]],
            [t[2], 0, -t[0]],
            [-t[1], t[0], 0]
        ], dtype=np.float32)
        
        # Essential matrix
        E = t_x @ R
        
        # Fundamental matrix
        K_inv = np.linalg.inv(K)
        F = K_inv.T @ E @ K_inv
        
        # Normalize by Frobenius norm (avoids division by zero when F[2,2] ≈ 0)
        F = F / np.linalg.norm(F)
        
        return F
    
    def compute_essential_matrix(self, frame_idx: int = 0) -> np.ndarray:
        """Compute essential matrix for a specific frame."""
        T_cam0 = self.get_frame_extrinsic(frame_idx, 0)
        T_cam1 = self.get_frame_extrinsic(frame_idx, 1)
        
        if T_cam0 is None or T_cam1 is None:
            T_cam0 = self.camera_params["Camera_0"]["T_world_to_cam_ref"]
            T_cam1 = self.camera_params["Camera_1"]["T_world_to_cam_ref"]
        
        T_cam0_to_world = np.linalg.inv(T_cam0)
        T_cam0_to_cam1 = T_cam1 @ T_cam0_to_world
        
        R = T_cam0_to_cam1[:3, :3]
        t = T_cam0_to_cam1[:3, 3]
        
        t_x = np.array([
            [0, -t[2], t[1]],
            [t[2], 0, -t[0]],
            [-t[1], t[0], 0]
        ], dtype=np.float32)
        
        E = t_x @ R
        E = E / np.linalg.norm(E)
        
        return E

    def _parse_flow_png(self, flow_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Parse optical flow from PNG file.
        
        VirtualKITTI2 Flow Encoding (as per dataset specification):
        - R channel: flow_x normalized by image width, quantized to [0; 2^16-1]
        - G channel: flow_y normalized by image height, quantized to [0; 2^16-1]
        - B channel: 0 for invalid flow (e.g., sky pixels), 1 for valid flow
        
        Denormalization: Converts from normalized [-1, 1] to pixel displacement.
        Note: The multiplication by (w-1) and (h-1) is used to match array indexing.
        """
        if not flow_path.exists():
            return None
        
        flow_img = cv2.imread(str(flow_path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)  # (h, w, c) uint16
        if flow_img is None:
            return None

        h, w, _c = flow_img.shape
        assert flow_img.dtype == np.uint16 and _c == 3
        
        # B channel: invalid flow flag (0 = invalid, 1 = valid)
        # In OpenCV BGR format: flow_img[..., 0] is B
        invalid = flow_img[..., 0] == 0
        
        # R, G channels: flow_x, flow_y normalized by image dimensions
        # In OpenCV BGR format: flow_img[..., 2] is R (flow_x), flow_img[..., 1] is G (flow_y)
        # Extract [R, G] to get [flow_x, flow_y]
        out_flow = 2.0 / (2**16 - 1.0) * flow_img[..., 2:0:-1].astype('f4') - 1 # float32
        
        # Denormalize from [-1, 1] to pixel space
        # flow_x: normalized value * (w - 1) gives pixel displacement in x direction
        # flow_y: normalized value * (h - 1) gives pixel displacement in y direction
        out_flow[..., 0] *= w - 1
        out_flow[..., 1] *= h - 1
        
        # Set invalid flow pixels to 0
        out_flow[invalid] = 0

        return out_flow, ~invalid

    def _parse_depth_png(self, depth_path: Path) -> np.ndarray:
        """Parse depth from PNG file.
        
        VirtualKITTI2 Depth Encoding (as per dataset specification):
        - 16-bit grayscale PNG (uint16)
        - Pixel intensity directly represents depth in centimeters
        - Encoding: 1 pixel value = 1 cm distance
        - Range: [0, 2^16-1] representing [0, 655.35 meters]
        - Far plane: 655.35 meters (pixels farther away are clipped)
        
        Processing:
        1. Load as uint16 using official one-liner: cv2.imread(..., cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        2. Convert from centimeters to meters by dividing by 100
        3. Mark pixels with depth=0 as NaN (invalid/no-data regions)
           - Depth 0 cm (0 meters) represents either camera plane or no-data
           - These regions are typically sky or out-of-bounds areas
        """
        if not depth_path.exists():
            return None
        
        # Load depth image using the official VirtualKITTI2 one-liner
        depth_img = cv2.imread(str(depth_path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)  # (h, w) uint16
        if depth_img is None:
            return None
        
        # Convert from centimeters (pixel values) to meters
        depth_meters = depth_img.astype(np.float32) / 100.0
        
        # Mark pixels with depth=0 as invalid/no-data
        # These typically represent sky regions or out-of-bounds areas
        depth_meters[depth_meters == 0] = np.nan
        
        return depth_meters
    
    def get_frame(self, frame_idx: int, camera: int = 0) -> Dict[str, np.ndarray]:
        """
        Load all data for a specific frame and camera.
        
        Args:
            frame_idx: Frame index in the dataset (0-based for extrinsics)
            camera: 0 for left camera, 1 for right camera
            
        Returns:
            Dictionary containing all loaded data for the frame
        """
        # Convert dataset frame index to file index
        if frame_idx in self.frame_indices:
            file_idx = self.file_indices[frame_idx]
        else:
            # If mapping doesn't exist, assume they're the same
            file_idx = f'{frame_idx:05d}'
            if file_idx not in self.file_indices:
                raise ValueError(f"Frame {frame_idx} not available. Available frames: [0-{list[self.frame_indices][-1]}]...")
        
        camera_str = f"Camera_{camera}"
        data = {}
        
        # Load RGB image
        rgb_path = self.scene_dir / self.file_templates['rgb'].format(camera, file_idx)
        
        data["rgb"] = cv2.imread(str(rgb_path)) # (h, w, c) uint8 [0, 255]
        if data["rgb"] is None:
            raise ValueError(f"Could not load RGB image: {rgb_path}")
        data["rgb"] = cv2.cvtColor(data["rgb"], cv2.COLOR_BGR2RGB) # (h, w, c)
        
        # Load depth
        depth_path = self.scene_dir / self.file_templates['depth'].format(camera, file_idx)
        data["depth"] = self._parse_depth_png(depth_path) # float32
        
        # Load optical flows
        flow_fwd_path = self.scene_dir / self.file_templates["forward_flow"].format(camera, file_idx)
        flow_data = self._parse_flow_png(flow_fwd_path) # float32, bool
        if flow_data is not None:
            data["flow_forward"], data["flow_forward_mask"] = flow_data
        
        flow_bwd_path = self.scene_dir / self.file_templates["backward_flow"].format(camera, file_idx)
        flow_data = self._parse_flow_png(flow_bwd_path) # float32, bool
        if flow_data is not None:
            data["flow_backward"], data["flow_backward_mask"] = flow_data
        
        # ============================
        # CAMERA PARAMETERS (frame-specific when available)
        # ============================
        # Get frame-specific extrinsic
        extrinsic = self.get_frame_extrinsic(frame_idx, camera)
        
        if extrinsic is not None:
            # Use frame-specific extrinsic
            data["extrinsic"] = extrinsic
            data["rotation"] = extrinsic[:3, :3]
            data["translation"] = extrinsic[:3, 3]
            data["projection"] = self.camera_params[camera_str]["intrinsic"] @ extrinsic[:3, :]
        else:
            # Fall back to reference extrinsic
            data["extrinsic"] = self.camera_params[camera_str]["T_world_to_cam_ref"]
            data["rotation"] = self.camera_params[camera_str]["rotation_ref"]
            data["translation"] = self.camera_params[camera_str]["translation_ref"]
            data["projection"] = self.camera_params[camera_str]["projection_ref"]
        
        # Common parameters
        data["intrinsic"] = self.camera_params[camera_str]["intrinsic"].copy()
        data["projection_rect"] = self.camera_params[camera_str]["projection_rect"].copy()
        
        # Add frame metadata
        data["frame_idx"] = frame_idx
        data["file_idx"] = file_idx
        data["camera"] = camera
        data["scene"] = self.scene
        data["variation"] = self.variation
        
        return data
    
    def get_stereo_pair(self, frame_idx: int) -> Dict[str, Dict]:
        """
        Load stereo pair (left and right camera) for a frame.
        """
        left_data = self.get_frame(frame_idx, camera=0)
        right_data = self.get_frame(frame_idx, camera=1)
        
        # Get extrinsics for this frame
        T_left = left_data["extrinsic"]
        T_right = right_data["extrinsic"]
        
        # Compute relative transformation for this frame
        T_left_to_world = np.linalg.inv(T_left)
        T_left_to_right = T_right @ T_left_to_world
        
        R_left_to_right = T_left_to_right[:3, :3]
        t_left_to_right = T_left_to_right[:3, 3]
        baseline = np.linalg.norm(t_left_to_right)
        
        # Add stereo-specific information
        stereo_data = {
            "left": left_data,
            "right": right_data,
            "frame_idx": frame_idx,
            "baseline": baseline,
            "R_left_to_right": R_left_to_right,
            "t_left_to_right": t_left_to_right,
            "T_left_to_right": T_left_to_right,
            "fundamental_matrix": self.compute_fundamental_matrix(frame_idx),
            "essential_matrix": self.compute_essential_matrix(frame_idx)
        }
        
        return stereo_data
    
    def compute_disparity_from_depth(self, depth: np.ndarray, frame_idx: int = 0) -> np.ndarray:
        """Compute disparity map from depth map."""
        # Use frame-specific baseline if available
        stereo_info = self.get_stereo_pair(frame_idx)
        baseline = stereo_info["baseline"]
        
        K = self.camera_params["Camera_0"]["intrinsic"]
        focal_length = K[0, 0]
        
        disparity = (focal_length * baseline) / depth
        disparity[np.isnan(disparity) | np.isinf(disparity) | (disparity < 0)] = 0
        
        return disparity
    
    def project_3d_to_pixel(self, points_3d: np.ndarray, frame_idx: int = 0, camera: int = 0) -> np.ndarray:
        """Project 3D world points to pixel coordinates for specific frame."""
        frame_data = self.get_frame(frame_idx, camera)
        
        K = frame_data["intrinsic"]
        R = frame_data["rotation"]
        t = frame_data["translation"]
        
        # Convert to homogeneous coordinates
        points_3d_homo = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
        
        # Transform to camera coordinates
        points_cam = (R @ points_3d.T + t.reshape(3, 1)).T
        
        # Project to image plane
        points_img_homo = (K @ points_cam.T).T
        
        # Convert to pixel coordinates
        points_pixel = points_img_homo[:, :2] / points_img_homo[:, 2:3]
        
        return points_pixel
    
    def __len__(self) -> int:
        """Return the number of frames in the dataset."""
        return len(self.file_indices)
    
    def get_all_frames(self, camera: int = 0) -> List[Dict]:
        """Load all frames for a camera."""
        frames = []
        for i in self.frame_indices:
            try:
                frames.append(self.get_frame(i, camera))
            except ValueError as e:
                print(f"Warning: Could not load frame {i}: {e}")
                continue
        return frames
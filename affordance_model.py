from typing import Tuple, Optional, Dict

import numpy as np
from matplotlib import cm
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage

from common import draw_grasp

'''
problem 4: improvement on finer grasp angle
'''
default_angle = 22.5
improved_angle = 11.25

def get_gaussian_scoremap(
        shape: Tuple[int, int], 
        keypoint: np.ndarray, 
        sigma: float=1, dtype=np.float32) -> np.ndarray:
    """
    Generate a image of shape=:shape:, generate a Gaussian distribtuion
    centered at :keypont: with standard deviation :sigma: pixels.
    keypoint: shape=(2,)
    """
    coord_img = np.moveaxis(np.indices(shape),0,-1).astype(dtype)
    sqrt_dist_img = np.square(np.linalg.norm(
        coord_img - keypoint[::-1].astype(dtype), axis=-1))
    scoremap = np.exp(-0.5/np.square(sigma)*sqrt_dist_img)
    return scoremap

class AffordanceDataset(Dataset):
    """
    Transformational dataset.
    raw_dataset is of type train.RGBDataset
    """
    def __init__(self, raw_dataset: Dataset):
        super().__init__()
        self.raw_dataset = raw_dataset
    
    def __len__(self) -> int:
        return len(self.raw_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Transform the raw RGB dataset element into
        training targets for AffordanceModel.
        return: 
        {
            'input': torch.Tensor (3,H,W), torch.float32, range [0,1]
            'target': torch.Tensor (1,H,W), torch.float32, range [0,1]
        }
        Note: self.raw_dataset[idx]['rgb'] is torch.Tensor (H,W,3) torch.uint8
        """
        # checkout train.RGBDataset for the content of data
        data = self.raw_dataset[idx]
        # TODO: complete this method
        # Hint: Use get_gaussian_scoremap
        # Hint: https://imgaug.readthedocs.io/en/latest/source/examples_keypoints.html
        # ===============================================================================
        train_data = dict()
        rgb = data['rgb'].numpy()
        center_point = data['center_point']
        angle = np.float32(data['angle'])
        img_shape = rgb.shape[:2]
        
        kps = KeypointsOnImage([
                Keypoint(x=center_point[0], y=center_point[1])
            ], shape=tuple(img_shape))
        
        seq = iaa.Sequential([
            iaa.Affine(
                rotate=-angle
            )
        ])
        # Augment keypoints and images
        image_aug, kps_aug = seq(image=rgb, keypoints=kps)
        # get input and normalize so range = [0, 1]
        image_aug = np.moveaxis(image_aug, -1, 0)
        image_aug = torch.squeeze(torch.div(torch.from_numpy(image_aug).to(torch.float32), 255))
        train_data['input'] = image_aug
        
        m = get_gaussian_scoremap(img_shape, kps_aug[0].xy)
        # get a new tensor with a dimension of size one and store it as target
        m_t = torch.from_numpy(m).to(torch.float32).unsqueeze(0)
        train_data['target'] = m_t
        
        return train_data
        # ===============================================================================


class AffordanceModel(nn.Module):
    def __init__(self, n_channels: int=3, n_classes: int=1, **kwargs):
        """
        A simplified U-Net with twice of down/up sampling and single convolution.
        ref: https://arxiv.org/abs/1505.04597, https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
        :param n_channels (int): number of channels (for grayscale 1, for rgb 3)
        :param n_classes (int): number of segmentation classes (num objects + 1 for background)
        """
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = nn.Sequential(
            # For simplicity:
            #     use padding 1 for 3*3 conv to keep the same Width and Height and ease concatenation
            nn.Conv2d(in_channels=n_channels, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential( 
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.outc = nn.Conv2d(in_channels=64, out_channels=n_classes, kernel_size=1)
        # hack to get model device
        self.dummy_param = nn.Parameter(torch.empty(0))

    @property
    def device(self) -> torch.device:
        return self.dummy_param.device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_inc = self.inc(x)  # N * 64 * H * W
        x_down1 = self.down1(x_inc)  # N * 128 * H/2 * W/2
        x_down2 = self.down2(x_down1)  # N * 256 * H/4 * W/4
        x_up1 = self.upconv1(x_down2)  # N * 128 * H/2 * W/2
        x_up1 = torch.cat([x_up1, x_down1], dim=1)  # N * 256 * H/2 * W/2
        x_up1 = self.conv1(x_up1)  # N * 128 * H/2 * W/2
        x_up2 = self.upconv2(x_up1)  # N * 64 * H * W
        x_up2 = torch.cat([x_up2, x_inc], dim=1)  # N * 128 * H * W
        x_up2 = self.conv2(x_up2)  # N * 64 * H * W
        x_outc = self.outc(x_up2)  # N * n_classes * H * W
        return x_outc

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict affordance using this model.
        This is required due to BCEWithLogitsLoss
        """
        return torch.sigmoid(self.forward(x))

    @staticmethod
    def get_criterion() -> torch.nn.Module:
        """
        Return the Loss object needed for training.
        Hint: why does nn.BCELoss does not work well?
        """
        return nn.BCEWithLogitsLoss()

    @staticmethod
    def visualize(input: np.ndarray, output: np.ndarray, 
            target: Optional[np.ndarray]=None) -> np.ndarray:
        """
        Visualize rgb input and affordance as a single rgb image.
        """
        cmap = cm.get_cmap('viridis')
        in_img = np.moveaxis(input, 0, -1)
        pred_img = cmap(output[0])[...,:3]
        row = [in_img, pred_img]
        if target is not None:
            gt_img = cmap(target[0])[...,:3]
            row.append(gt_img)
        img = (np.concatenate(row, axis=1)*255).astype(np.uint8)
        return img


    def predict_grasp(self, rgb_obs: np.ndarray
            ) -> Tuple[Tuple[int, int], float, np.ndarray]:
        """
        Given a RGB image observation, predict the grasping location and angle in image space.
        return coord, angle, vis_img
        :coord: tuple(int x, int y). By OpenCV convension, x is left-to-right and y is top-to-bottom.
        :angle: float. By OpenCV convension, angle is clockwise rotation.
        :vis_img: np.ndarray(shape=(H,W,3), dtype=np.uint8). Visualize prediction as a RGB image.

        Note: torchvision's rotation is counter clockwise, while imgaug,OpenCV's rotation are clockwise.
        """
        device = self.device
        # TODO: complete this method (prediction)
        # Hint: why do we provide the model's device here? --> send the observation stack to device
        # make observation stack, rgb_obs: Numpy array [Height x Width x 3]
        img_shape = rgb_obs.shape[:2]
        rotated_rgb_stack = []
        '''
        problem 4: improvement on finer grasp angle
        8 --> 16: smaller bins
        '''
        for i in range(0, 16):
            rotate = iaa.Affine(rotate=(i*improved_angle))
            rotate_img = rotate(image=rgb_obs)
            # rgb_obs: (H, W, 3) --> (3, H, W)
            rotate_img = np.moveaxis(rotate_img, -1, 0)
            rotate_img = torch.div(torch.from_numpy(rotate_img).to(torch.float32), 255)
            rotate_img = torch.squeeze(rotate_img)
            rotated_rgb_stack.append(rotate_img)
        rotated_rgb_stack = torch.stack(rotated_rgb_stack)
        # send stack to device
        rotated_rgb_stack.to(device)

        # predict
        predicted = self.predict(rotated_rgb_stack)
        max_idx = int(torch.argmax(predicted))
        i = max_idx // (128*128)        # find which angle idx produced from argmax idx
        x = max_idx % (128 * 128) % 128  # find width
        y = max_idx % (128 * 128) // 128  # find height

        # rotate the keypoints back by the angle found
        kps = KeypointsOnImage([
            Keypoint(x=x, y=y)
        ], shape=tuple(img_shape))
        rotate_back = iaa.Affine(rotate=(-i * improved_angle))
        rotate_back_img, rotate_back_kps = rotate_back(image=rgb_obs, keypoints=kps)

        # extract coord from keypoints
        coord = rotate_back_kps[0].xy
        coord_tuple = (int(coord[0]), int(coord[1]))
        angle_recovery = -i * improved_angle
        # use common.draw_grasp with angle 0, as the image/keypoints has already been rotated
        draw_grasp(rgb_obs, coord_tuple, 0)
        # rotate the best/predicted rgb image
        # ===============================================================================
        # TODO: complete this method (visualization)
        # :vis_img: np.ndarray(shape=(H,W,3), dtype=np.uint8). Visualize prediction as a RGB image.
        # Hint: use common.draw_grasp
        # Hint: see self.visualize
        # Hint: draw a grey (127,127,127) line on the bottom row of each image.
        # ===============================================================================
        # rotate the best/predicted rgb image
        best_img = iaa.Affine(rotate=(improved_angle * i))(image=rgb_obs)
        vis_img = []
        for k in range(16):
            if i == k:
                vis_img.append(self.visualize(np.moveaxis(best_img, -1, 0)/255, predicted[i].detach().numpy()))
            else:
                vis_img.append(self.visualize(rotated_rgb_stack.numpy()[i], predicted[i].detach().numpy()))
        vis_img = np.vstack(vis_img)
        # ===============================================================================
        return coord_tuple, angle_recovery, vis_img


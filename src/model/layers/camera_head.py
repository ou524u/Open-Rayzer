# Adapted from https://github.com/yyfz/Pi3/blob/main/pi3/models/layers/camera_head.py

import torch
import torch.nn as nn
from copy import deepcopy
import torch.nn.functional as F

# code adapted from 'https://github.com/nianticlabs/marepo/blob/9a45e2bb07e5bb8cb997620088d352b439b13e0e/transformer/transformer.py#L172'
class ResConvBlock(nn.Module):
    """
    1x1 convolution residual block
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.head_skip = nn.Identity() if self.in_channels == self.out_channels else nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0)
        # self.res_conv1 = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0)
        # self.res_conv2 = nn.Conv2d(self.out_channels, self.out_channels, 1, 1, 0)
        # self.res_conv3 = nn.Conv2d(self.out_channels, self.out_channels, 1, 1, 0)

        # change 1x1 convolution to linear
        self.res_conv1 = nn.Linear(self.in_channels, self.out_channels)
        self.res_conv2 = nn.Linear(self.out_channels, self.out_channels)
        self.res_conv3 = nn.Linear(self.out_channels, self.out_channels)

    def forward(self, res):
        x = F.relu(self.res_conv1(res))
        x = F.relu(self.res_conv2(x))
        x = F.relu(self.res_conv3(x))
        res = self.head_skip(res) + x
        return res

class CameraHead(nn.Module): # input [b, n, d] output [b, 4, 4]
    def __init__(self, dim=512, image_size=256, patch_size=16, predict_fov=False):
        super().__init__()

        self.patch_h = image_size // patch_size
        self.patch_w = image_size // patch_size

        output_dim = dim
        self.res_conv = nn.ModuleList([deepcopy(ResConvBlock(output_dim, output_dim)) 
                for _ in range(2)])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.more_mlps = nn.Sequential(
            nn.Linear(output_dim,output_dim),
            nn.ReLU(),
            nn.Linear(output_dim,output_dim),
            nn.ReLU()
            )
        self.fc_t = nn.Linear(output_dim, 3)
        self.fc_rot = nn.Linear(output_dim, 9)

        self.predict_fov = predict_fov
        if predict_fov:
            self.fov_mlp = nn.Sequential(
                nn.Linear(output_dim,output_dim),
                nn.ReLU(),
                nn.Linear(output_dim,output_dim),
                nn.ReLU()
                )
            self.fov_head = nn.Linear(output_dim, 1)

    def forward(self, feat, patch_h=None, patch_w=None): 
        # we assue that a scene has only one intrinsics.

        restore_v = False
        if feat.dim() == 4:
            b, v, n, d = feat.shape
            feat = feat.reshape(b * v, n, d)
            restore_v = True

        b_v, _, _ = feat.shape

        if patch_h is None:
            patch_h = self.patch_h
        if patch_w is None:
            patch_w = self.patch_w

        for i in range(2):
            feat = self.res_conv[i](feat)

        # feat = self.avgpool(feat)
        feat = self.avgpool(feat.permute(0, 2, 1).reshape(b_v, -1, patch_h, patch_w).contiguous())              ##########
        feat = feat.view(feat.size(0), -1) # [b*v, d]

        fov = None
        if self.predict_fov:
            if restore_v:
                fov_feat = feat.view(b, v, -1).mean(dim=1) # [b, d]
            else:
                fov_feat = feat # [b, d]
            fov_feat = self.fov_mlp(fov_feat)
            fov = self.fov_head(fov_feat)
            fov = F.softplus(fov).clamp(min=1e-6) # make sure fov is positive

        feat = self.more_mlps(feat)  # [B, D_]
        with torch.amp.autocast(device_type='cuda', enabled=False):
            out_t = self.fc_t(feat.float())  # [B,3]
            out_r = self.fc_rot(feat.float())  # [B,9]
            pose = self.convert_pose_to_4x4(b_v, out_r, out_t, feat.device)

        if restore_v:
            pose = pose.view(b, v, 4, 4)

        return pose, fov # pose could be [b, v, 4, 4] or [b, 4, 4]. while fov be only [b, 1]

    def convert_pose_to_4x4(self, B, out_r, out_t, device):
        out_r = self.svd_orthogonalize(out_r)  # [N,3,3]
        pose = torch.zeros((B, 4, 4), device=device)
        pose[:, :3, :3] = out_r
        pose[:, :3, 3] = out_t
        pose[:, 3, 3] = 1.
        return pose

    def svd_orthogonalize(self, m):
        """Convert 9D representation to SO(3) using SVD orthogonalization.

        Args:
          m: [BATCH, 3, 3] 3x3 matrices.

        Returns:
          [BATCH, 3, 3] SO(3) rotation matrices.
        """
        if m.dim() < 3:
            m = m.reshape((-1, 3, 3))
        m_transpose = torch.transpose(torch.nn.functional.normalize(m, p=2, dim=-1), dim0=-1, dim1=-2)
        u, s, v = torch.svd(m_transpose)
        det = torch.det(torch.matmul(v, u.transpose(-2, -1)))
        # Check orientation reflection.
        r = torch.matmul(
            torch.cat([v[:, :, :-1], v[:, :, -1:] * det.view(-1, 1, 1)], dim=2),
            u.transpose(-2, -1)
        )
        return r
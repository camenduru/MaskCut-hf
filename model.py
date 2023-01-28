# This file is adapted from https://github.com/facebookresearch/CutLER/blob/077938c626341723050a1971107af552a6ca6697/maskcut/demo.py
# The original license file is the file named LICENSE.CutLER in this repo.

import sys

import numpy as np
import PIL.Image as Image
import torch
from scipy import ndimage

sys.path.append('CutLER/maskcut/')
sys.path.append('CutLER/')
import dino
from colormap import random_color
from crf import densecrf
from maskcut import maskcut
from third_party.TokenCut.unsupervised_saliency_detection import metric


def vis_mask(input, mask, mask_color):
    fg = mask > 0.5
    rgb = np.copy(input)
    rgb[fg] = (rgb[fg] * 0.3 + np.array(mask_color) * 0.7).astype(np.uint8)
    return Image.fromarray(rgb)


class Model:
    def __init__(self):
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.backbone = self.load_backbone()

    def load_backbone(self):
        # DINO hyperparameters
        vit_arch = 'base'
        vit_feat = 'k'
        patch_size = 8
        # DINO pre-trained model
        url = 'https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth'
        feat_dim = 768

        # extract patch features with a pretrained DINO model
        backbone = dino.ViTFeat(url, feat_dim, vit_arch, vit_feat, patch_size)
        backbone.eval()
        backbone.to(self.device)
        return backbone

    def __call__(self, img_path, tau, n, fixed_size=480):
        # get pseudo-masks with MaskCut
        bipartitions, _, I_new = maskcut(img_path,
                                         self.backbone,
                                         self.backbone.patch_size,
                                         tau,
                                         N=n,
                                         fixed_size=fixed_size,
                                         cpu=self.device.type == 'cpu')
        I = Image.open(img_path).convert('RGB')
        width, height = I.size
        pseudo_mask_list = []
        for idx, bipartition in enumerate(bipartitions):
            # post-process pseudo-masks with CRF
            pseudo_mask = densecrf(np.array(I_new), bipartition)
            pseudo_mask = ndimage.binary_fill_holes(pseudo_mask >= 0.5)

            # filter out the mask that have a very different pseudo-mask after the CRF
            mask1 = torch.from_numpy(bipartition).to(self.device)
            mask2 = torch.from_numpy(pseudo_mask).to(self.device)
            if metric.IoU(mask1, mask2) < 0.5:
                pseudo_mask = pseudo_mask * -1

            # construct binary pseudo-masks
            pseudo_mask[pseudo_mask < 0] = 0
            pseudo_mask = Image.fromarray(np.uint8(pseudo_mask * 255))
            pseudo_mask = np.asarray(pseudo_mask.resize((width, height)))

            pseudo_mask = pseudo_mask.astype(np.uint8)
            upper = np.max(pseudo_mask)
            lower = np.min(pseudo_mask)
            thresh = upper / 2.0
            pseudo_mask[pseudo_mask > thresh] = upper
            pseudo_mask[pseudo_mask <= thresh] = lower
            pseudo_mask_list.append(pseudo_mask)
        return pseudo_mask_list

import argparse
import cv2
import glob
import numpy as np
import os
import torch
from basicsr.utils import imwrite
from gfpgan import GFPGANer

def enhance_image_with_gfpgan(input_path, output_path, version='1.3', upscale=2, bg_upsampler='realesrgan', weight=0.5):
    """
    Enhance an image using GFPGAN.

    Args:
        input_path (str): Path to the input image or folder.
        output_path (str): Path to the output folder.
        version (str): GFPGAN model version. Default: '1.3'.
        upscale (int): Final upsampling scale of the image. Default: 2.
        bg_upsampler (str): Background upsampler. Default: 'realesrgan'.
        weight (float): Adjustable weights. Default: 0.5.
    """
    # Set up GFPGAN restorer
    if version == '1':
        arch = 'original'
        channel_multiplier = 1
        model_name = 'GFPGANv1'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth'
    elif version == '1.2':
        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANCleanv1-NoCE-C2'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth'
    elif version == '1.3':
        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANv1.3'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth'
    elif version == '1.4':
        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANv1.4'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
    elif version == 'RestoreFormer':
        arch = 'RestoreFormer'
        channel_multiplier = 2
        model_name = 'RestoreFormer'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth'
    else:
        raise ValueError(f'Wrong model version {version}.')

    # Determine model paths
    model_path = os.path.join('experiments/pretrained_models', model_name + '.pth')
    if not os.path.isfile(model_path):
        model_path = os.path.join('gfpgan/weights', model_name + '.pth')
    if not os.path.isfile(model_path):
        # Download pre-trained models from URL
        model_path = url

    # Set up background upsampler
    if bg_upsampler == 'realesrgan':
        if not torch.cuda.is_available():  # CPU
            bg_upsampler = None
        else:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            bg_upsampler = RealESRGANer(
                scale=2,
                model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                model=model,
                tile=400,
                tile_pad=10,
                pre_pad=0,
                half=True
            )
    else:
        bg_upsampler = None

    # Initialize GFPGAN restorer
    restorer = GFPGANer(
        model_path=model_path,
        upscale=upscale,
        arch=arch,
        channel_multiplier=channel_multiplier,
        bg_upsampler=bg_upsampler
    )

    # Process the input image
    input_img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    _, _, restored_img = restorer.enhance(
        input_img,
        has_aligned=False,
        only_center_face=False,
        paste_back=True,
        weight=weight
    )

    # Save the restored image
    os.makedirs(output_path, exist_ok=True)
    output_image_path = os.path.join(output_path, os.path.basename(input_path))
    imwrite(restored_img, output_image_path)

    return output_image_path
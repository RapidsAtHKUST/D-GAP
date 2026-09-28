import torch
import numpy as np
from PIL import Image
import os

from utils import isin
from data_augmentation.utils import create_mask_from_bboxes
from data_augmentation.utils import *
from data_augmentation.frequency import *
from torch.fft import fft2, ifft2, fftshift, ifftshift

import random




import numpy as np
import os
import torch
from torchvision import transforms
from PIL import Image, ImageFilter
from scipy.ndimage import gaussian_filter
from math import sqrt


# NOTE:
# This is a rewritten version of the original CopyPasteAugment class.
# The current implementation focuses on low-frequency amplitude mixing
# in the Fourier domain: a shared random coefficient λ ~ U(0, alpha)
# is used to mix the low-frequency magnitude components of two images
# (while keeping their phases), and the same λ is applied uniformly
# across channels and within the selected low-frequency window.

class CopyPasteAugment:
    def __init__(self, labeled_dataset, same_y_observed=False, translate_upper=None, translate_lower=None, 
                 resize_upper=None, resize_lower=None, cutoff_frequency=1, device='cuda', save_dir='egs'):
        """
        Initialize the augmentor for Copy-Paste augmentation.
        :param labeled_dataset: Dataset containing images and labels
        :param translate_upper, translate_lower: Translation augmentation parameters
        :param resize_upper, resize_lower: Resize augmentation parameters
        :param cutoff_frequency: Cutoff frequency for high-pass and low-pass filtering
        :param device: Device to run the computations on (default is 'cuda')
        :param save_dir: Directory to save augmented images (optional)
        """
        assert not (translate_upper is None) ^ (translate_lower is None)
        assert not (resize_upper is None) ^ (resize_lower is None)

        self.translate_range = (translate_lower, translate_upper) if translate_upper is not None else None
        self.resize_range = (resize_lower, resize_upper) if resize_upper is not None else None
        self.dataset = labeled_dataset
        self.device = device
        self.save_dir = save_dir
        self.ix = 1

        # Get indices of samples grouped by hospital
        self.hospital_indices = self._get_hospital_indices()

        if self.save_dir and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def _get_hospital_indices(self):
        """
        Generate or retrieve the list of indices for samples from different hospitals.
        In this case, the hospital ID is retrieved from the metadata array.
        """
        hospital_indices = {}
        for idx in range(len(self.dataset)):
            # The hospital (center) information is stored in the first column of the metadata
            hospital_id = self.dataset._metadata_array[idx, 0].item()  # Access hospital ID from metadata array
            if hospital_id not in hospital_indices:
                hospital_indices[hospital_id] = []
            hospital_indices[hospital_id].append(idx)
        return hospital_indices

    def __call__(self, img):
        """
        Perform Copy-Paste augmentation by adding parts from images of the same or different hospitals.
        :param img: The input image to augment
        :param ix: Index of the image in the dataset
        :return: Augmented image
        """

        self.ix = self.ix + 1
        y = self.dataset._y_array[self.ix].item()  # Get the class label of the current image

        # Get a background image from a different hospital sample
        bg_img = self.get_empty_img_from_different_hospital(self.ix)
        bg_img = bg_img.copy().resize(img.size)


        # Apply Copy-Paste augmentation: paste foreground image onto background
        alpha = 0.9 # random.uniform(0.9, 0.99)
        combined_img, _ = self.colorful_spectrum_mix_cuda1(img, bg_img, alpha=alpha)

        return combined_img

    def get_empty_img_from_different_hospital(self, ix):
        """ 
        Randomly select a sample from a different hospital to use as background.
        """
        current_hospital_id = self.dataset._metadata_array[ix, 0].item()  # Get current sample's hospital ID
        possible_hospitals = [hospital for hospital in self.hospital_indices if hospital != current_hospital_id]
        
        if not possible_hospitals:
            return self.dataset.get_input(ix)  # Fallback to the same hospital if no other options
        
        # Randomly choose a different hospital
        selected_hospital = np.random.choice(possible_hospitals)
        empty_ix = np.random.choice(self.hospital_indices[selected_hospital])
        
        return self.dataset.get_input(empty_ix)  # Get the image from the selected hospital

    def colorful_spectrum_mix_cuda1(self, img1, img2, alpha, ratio=1.0):
        """
        Perform spectrum mixing on two images using CUDA acceleration.
        :param img1: First image (PIL.Image)
        :param img2: Second image (PIL.Image)
        :param alpha: Mixing coefficient
        :param ratio: Ratio used to define the cropped low-frequency window
        :return: Two mixed images as PIL.Image
        """
        lam = np.random.uniform(0, alpha)

        img1 = torch.tensor(np.array(img1), dtype=torch.float32, device='cuda')
        img2 = torch.tensor(np.array(img2), dtype=torch.float32, device='cuda')

        assert img1.shape == img2.shape
        h, w, c = img1.shape
        h_crop = int(h * np.sqrt(ratio))
        w_crop = int(w * np.sqrt(ratio))
        h_start = h // 2 - h_crop // 2
        w_start = w // 2 - w_crop // 2

        # FFT and spectrum shifting
        img1_fft = fft2(img1, dim=(0, 1))
        img2_fft = fft2(img2, dim=(0, 1))
        img1_abs, img1_pha = img1_fft.abs(), img1_fft.angle()
        img2_abs, img2_pha = img2_fft.abs(), img2_fft.angle()

        img1_abs = fftshift(img1_abs, dim=(0, 1))
        img2_abs = fftshift(img2_abs, dim=(0, 1))

        img1_abs_ = img1_abs.clone()
        img2_abs_ = img2_abs.clone()

        # Mix spectra in the cropped low-frequency region
        img1_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = lam * img2_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img1_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop]
        img2_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = lam * img1_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img2_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop]

        img1_abs = ifftshift(img1_abs, dim=(0, 1))
        img2_abs = ifftshift(img2_abs, dim=(0, 1))

        # IFFT and image reconstruction
        img21 = img1_abs * torch.exp(1j * img1_pha)
        img12 = img2_abs * torch.exp(1j * img2_pha)
        img21 = torch.real(ifft2(img21, dim=(0, 1)))
        img12 = torch.real(ifft2(img12, dim=(0, 1)))

        # Convert back to [0, 255] range and then to PIL images
        img21 = img21.cpu().numpy().clip(0, 255).astype(np.uint8)
        img12 = img12.cpu().numpy().clip(0, 255).astype(np.uint8)

        img21 = Image.fromarray(img21)
        img12 = Image.fromarray(img12)

        return img21, img12

    def colorful_spectrum_mix_cuda(self, img1, img2, alpha, ratio=1.0):
        """
        Perform spectrum mixing on two images and move the computation to GPU for acceleration.
        :param img1: First image (PyTorch tensor, shape: [H, W, C] or [C, H, W] depending on usage)
        :param img2: Second image (PyTorch tensor, shape: [H, W, C] or [C, H, W])
        :param alpha: Mixing coefficient
        :param ratio: Ratio used to define the cropped low-frequency window
        :return: Two mixed images (PyTorch tensors)
        """
        
        # Random mixing factor in [0, alpha]
        lam = torch.rand(1, device=self.device) * alpha

        assert img1.shape == img2.shape, "Two images must have the same size"
        h, w, c = img1.shape
        
        # Crop size for the low-frequency region
        h_crop = int(h * sqrt(ratio))  
        w_crop = int(w * sqrt(ratio))  
        h_start = h // 2 - h_crop // 2  
        w_start = w // 2 - w_crop // 2  

        # 2D Fourier transform using PyTorch FFT
        img1_fft = torch.fft.fft2(img1, dim=(0, 1))  
        img2_fft = torch.fft.fft2(img2, dim=(0, 1))

        # Magnitude and phase
        img1_abs = img1_fft.abs()
        img1_pha = img1_fft.angle()
        img2_abs = img2_fft.abs()
        img2_pha = img2_fft.angle()

        # Shift zero-frequency component to the center of the spectrum
        img1_abs = torch.fft.fftshift(img1_abs, dim=(0, 1))
        img2_abs = torch.fft.fftshift(img2_abs, dim=(0, 1))

        # Make copies of the magnitude for mixing
        img1_abs_ = img1_abs.clone()
        img2_abs_ = img2_abs.clone()

        img1_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
            lam * img2_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img1_abs_[
                                                                                              h_start:h_start + h_crop,
                                                                                              w_start:w_start + w_crop]
        img2_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
            lam * img1_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img2_abs_[
                                                                                              h_start:h_start + h_crop,
                                                                                              w_start:w_start + w_crop]
                                    
        # Shift the spectrum back
        img1_abs = torch.fft.ifftshift(img1_abs, dim=(0, 1))
        img2_abs = torch.fft.ifftshift(img2_abs, dim=(0, 1))

        # Inverse FFT and reconstruct images (take real part)
        img21 = img1_abs * torch.exp(1j * img1_pha)
        img12 = img2_abs * torch.exp(1j * img2_pha)

        img21 = torch.fft.ifft2(img21, dim=(0, 1)).real  # 取实部
        img12 = torch.fft.ifft2(img12, dim=(0, 1)).real  # 取实部

        # Clamp to valid range and convert to uint8
        img21 = torch.clamp(img21, 0, 255).to(torch.uint8)
        img12 = torch.clamp(img12, 0, 255).to(torch.uint8)

        return img21, img12


    def _save_image(self, img, ix, suffix):
        """ Save the augmented image with a suffix for identification """
        if self.save_dir:
            file_path = os.path.join(self.save_dir, f"{ix}_{suffix}.png")
            img.save(file_path)

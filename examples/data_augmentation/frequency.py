import torch
import torch.fft
from torchvision import transforms

class Frequency:
    def __init__(self, cutoff_frequency=5, device='cuda'):
        self.cutoff_frequency = cutoff_frequency
        self.device = device

    def gaussian_filter_low_pass(self, fshift):
        rows, cols = fshift.shape[-2], fshift.shape[-1]
        center_row, center_col = rows // 2, cols // 2
        
        # Create meshgrid for u and v
        u = torch.arange(0, rows, dtype=torch.float32, device=self.device)
        v = torch.arange(0, cols, dtype=torch.float32, device=self.device)
        
        u, v = torch.meshgrid(u, v)  # Create grid of u, v coordinates
        
        # Compute Euclidean distance matrix
        distance = torch.sqrt((u - center_row) ** 2 + (v - center_col) ** 2)
        
        # Create the low-pass filter mask
        mask = (distance <= self.cutoff_frequency).float()

        return fshift * mask

    def gaussian_filter_high_pass(self, fshift):
        rows, cols = fshift.shape[-2], fshift.shape[-1]
        center_row, center_col = rows // 2, cols // 2
        
        # Create meshgrid for u and v
        u = torch.arange(0, rows, dtype=torch.float32, device=self.device)
        v = torch.arange(0, cols, dtype=torch.float32, device=self.device)
        
        u, v = torch.meshgrid(u, v)  # Create grid of u, v coordinates
        
        # Compute Euclidean distance matrix
        distance = torch.sqrt((u - center_row) ** 2 + (v - center_col) ** 2)
        
        # Create the high-pass filter mask
        mask = (distance > self.cutoff_frequency).float()

        return fshift * mask

    def ifft(self, fshift):
        f_ishift = torch.fft.ifftshift(fshift)
        img_back = torch.abs(torch.fft.ifft2(f_ishift))
        return img_back

    def __call__(self, img, filter_type='high_pass'):
        # Convert PIL image to tensor and move to the chosen device (CUDA)
        img_tensor = transforms.ToTensor()(img).to(self.device)
        
        # Separate the channels
        r, g, b = img_tensor[0], img_tensor[1], img_tensor[2]

        # Perform Fourier transform on each channel
        f_r = torch.fft.fft2(r)
        f_g = torch.fft.fft2(g)
        f_b = torch.fft.fft2(b)

        # Center the Fourier transforms
        fshift_r = torch.fft.fftshift(f_r)
        fshift_g = torch.fft.fftshift(f_g)
        fshift_b = torch.fft.fftshift(f_b)

        # Apply low pass or high pass filter
        if filter_type == 'low_pass':
            fshift_r = self.gaussian_filter_low_pass(fshift_r)
            fshift_g = self.gaussian_filter_low_pass(fshift_g)
            fshift_b = self.gaussian_filter_low_pass(fshift_b)
        elif filter_type == 'high_pass':
            fshift_r = self.gaussian_filter_high_pass(fshift_r)
            fshift_g = self.gaussian_filter_high_pass(fshift_g)
            fshift_b = self.gaussian_filter_high_pass(fshift_b)

        # Perform inverse Fourier transform on each channel
        img_r = self.ifft(fshift_r)
        img_g = self.ifft(fshift_g)
        img_b = self.ifft(fshift_b)

        # Combine the channels
        img_filtered = torch.stack((img_r, img_g, img_b), dim=0)

        # Convert back to PIL Image
        img_filtered_pil = transforms.ToPILImage()(img_filtered.cpu())  # Move back to CPU for PIL
        # img_filtered_pil.save('high_pass_filtered_image.jpg')
        return img_filtered_pil

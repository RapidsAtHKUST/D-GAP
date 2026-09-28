import torch
from PIL import Image
from data_augmentation.utils import *
import torch.nn.functional as F
from torchvision.transforms import ToTensor, ToPILImage
import numpy as np


ACCEPTABLE_OVERLAP_THRESHOLD = 0.2
class RandomCrop:
    def __init__(self, size, labeled_dataset, num_tries=3, bbox_aware=False):
        assert size[0] > 0 and size[1] > 0
        self.width, self.height = size
        self.bbox_aware = bbox_aware
        self.num_tries = num_tries
        self.dataset = labeled_dataset

        self.nargs = 2

    def __call__(self, img, ix):
        # print(img.size)
        w, h = img.size
        # get bboxes if wanted and available
        if self.bbox_aware:
            try:
                bboxes, conf = self.dataset.get_bbox(ix)
                if conf < 0.5: bboxes = []
                # unnormalize bboxes
                bboxes = [xywh_to_xyxy(
                    bbox[0] * w,
                    bbox[1] * h,
                    bbox[2] * w,
                    bbox[3] * h,
                ) for bbox in bboxes]
            except:
                bboxes = None
        else:
            bboxes = None

        # set crop area
        if bboxes is not None and len(bboxes):
            tries = 1
            xy = sample_rectangle(w, h, self.width, self.height)
            while max([box_overlap(bbox, xy) for bbox in bboxes]) < ACCEPTABLE_OVERLAP_THRESHOLD and tries < self.num_tries:
                xy = sample_rectangle(w, h, self.width, self.height)
                tries += 1
        else:
            xy = sample_rectangle(w, h, self.width, self.height)
        
        # crop & return
        img = img.copy().crop(box=xywh_to_xyxy(*xy))
        # print(img.size)
        return img


# def high_pass(img_tensor, cutoff_freq=0.1):
#     """
#     使用FFT进行高通滤波
#     :param img_tensor: 输入的图像tensor，形状为 [C, H, W]
#     :param cutoff_freq: 高通滤波器的截止频率，范围从 0 到 1
#     :return: 高通滤波后的图像 tensor
#     """
#     # 获取图像的大小
#     c, h, w = img_tensor.shape
#     # 执行FFT变换
#     fft_img = torch.fft.fft2(img_tensor)

#     # 计算频率的中心
#     fx = torch.fft.fftfreq(w, d=1.0 / w).to(img_tensor.device)
#     fy = torch.fft.fftfreq(h, d=1.0 / h).to(img_tensor.device)
#     fx, fy = torch.meshgrid(fx, fy)

#     # 计算距离中心点的频率
#     dist = torch.sqrt(fx**2 + fy**2)

#     # 创建高通滤波器掩码
#     high_pass_mask = dist > cutoff_freq * dist.max()
    
#     # 应用高通滤波器掩码
#     fft_img = fft_img * high_pass_mask

#     # 执行逆FFT
#     img_filtered = torch.fft.ifft2(fft_img)
    
#     # 取实部并返回
#     return img_filtered.real

# # 假设已经有了高通滤波器的函数
# def high_pass_filter(img, cutoff_freq=0.1):
#     """
#     应用高通滤波器，假设图像为PIL图像
#     :param img: 输入的PIL图像
#     :param cutoff_freq: 高通滤波器的截止频率，范围从 0 到 1
#     :return: 高通滤波后的PIL图像
#     """
#     img_tensor = ToTensor()(img).unsqueeze(0).cuda()  # 转为Tensor，并移动到GPU
#     img_filtered = high_pass(img_tensor, cutoff_freq)  # 调用高通滤波器
#     return ToPILImage()(img_filtered.squeeze(0).cpu())  # 转回PIL图像

# class RandomCrop:
#     def __init__(self, size, labeled_dataset, num_tries=3, bbox_aware=False, apply_high_pass=True):
#         assert size[0] > 0 and size[1] > 0
#         self.width, self.height = size
#         self.bbox_aware = bbox_aware
#         self.num_tries = num_tries
#         self.dataset = labeled_dataset
#         self.apply_high_pass = apply_high_pass  # 是否应用高通滤波
#         self.nargs = 2

#     def __call__(self, img, ix):
#         w, h = img.size
        
#         # get bboxes if wanted and available
#         if self.bbox_aware:
#             try:
#                 bboxes, conf = self.dataset.get_bbox(ix)
#                 if conf < 0.5: bboxes = []
#                 # unnormalize bboxes
#                 bboxes = [xywh_to_xyxy(
#                     bbox[0] * w,
#                     bbox[1] * h,
#                     bbox[2] * w,
#                     bbox[3] * h,
#                 ) for bbox in bboxes]
#             except:
#                 bboxes = None
#         else:
#             bboxes = None

#         # set crop area
#         if bboxes is not None and len(bboxes):
#             tries = 1
#             xy = sample_rectangle(w, h, self.width, self.height)
#             while max([box_overlap(bbox, xy) for bbox in bboxes]) < ACCEPTABLE_OVERLAP_THRESHOLD and tries < self.num_tries:
#                 xy = sample_rectangle(w, h, self.width, self.height)
#                 tries += 1
#         else:
#             xy = sample_rectangle(w, h, self.width, self.height)
        
#         # crop & apply high-pass filter if needed
#         img = img.copy().crop(box=xywh_to_xyxy(*xy))

#         # 如果启用了高通滤波，应用高通滤波器
#         if self.apply_high_pass:
#             img = high_pass_filter(img, cutoff_freq=0.1)

#         return img
import torch
import numpy as np
from PIL import Image
import os

from utils import isin
from data_augmentation.utils import create_mask_from_bboxes
from data_augmentation.utils import *
from data_augmentation.frequency import *


# class CopyPasteAugment:
#     def __init__(self, labeled_dataset, same_y_observed=False, translate_upper=None, translate_lower=None, 
#                  resize_upper=None, resize_lower=None, cutoff_frequency=1, device='cuda', save_dir='egs'):
#         """
#         初始化函数，用于设定图像增强的参数并创建 Frequency 类的实例。
#         :param labeled_dataset: 数据集，包含图片和标签
#         :param translate_upper, translate_lower: 平移增强参数
#         :param resize_upper, resize_lower: 缩放增强参数
#         :param cutoff_frequency: 截止频率，控制高通低通滤波器的切换频率
#         :param device: 计算设备，默认为 'cuda'
#         :param save_dir: 保存图像的目录路径，默认为 None
#         """
#         assert labeled_dataset.dataset_name in ("iwildcam", "birdcalls")
#         assert hasattr(labeled_dataset, 'get_bbox'), "Dataset must have bbox or mask methods"
#         assert not (translate_upper is None) ^ (translate_lower is None)
#         assert not (resize_upper is None) ^ (resize_lower is None)

#         self.translate_range = (translate_lower, translate_upper) if translate_upper is not None else None
#         self.resize_range = (resize_lower, resize_upper) if resize_upper is not None else None
#         self.dataset = labeled_dataset
#         self.nargs = 2  # for the Transform class, to signal we want both img and ix passed to __call__

#         self.classes_to_not_augment = [0]  # Empty class for iwildcam, birdcalls
#         if 'train' in self.dataset.empty_indices:
#             self.empty_indices = self.dataset.empty_indices['train']
#         else:
#             self.empty_indices = np.array([], dtype=int)

#         # 初始化 Frequency 类
#         self.frequency = Frequency(cutoff_frequency=cutoff_frequency, device=device)
#         self.save_dir = save_dir

#         if self.save_dir and not os.path.exists(self.save_dir):
#             os.makedirs(self.save_dir)

#     def __call__(self, img, ix):
#         """
#         将当前图像的高频部分与空白图像的低频部分叠加。
#         """
#         dataset = self.dataset

#         # 不进行增强的类
#         y = dataset.y_array[ix].item()
#         if y in self.classes_to_not_augment: 
#             return img  # IDENTITY CASE

#         # 获取相关的注释信息
#         if hasattr(dataset, 'get_mask'):
#             mask = dataset.get_mask(ix, resize_wh=img.size)  # 获取图像遮罩
#         else:
#             mask = None

#         if mask is None or np.all(np.array(mask) == 0):  # 如果没有有效的遮罩
#             bboxes, conf = dataset.get_bbox(ix)
#             if conf < 0.5 or len(bboxes) == 0: 
#                 return img  # IDENTITY CASE
#             mask = create_mask_from_bboxes(bboxes, img.size)

#         # 获取空白背景图像
#         bg_img = self.get_empty_img()
#         bg_img = bg_img.copy().resize(img.size)

#         # 获取当前图像的高频部分（高通滤波）
#         high_freq_img = self.frequency(img, filter_type='high_pass')

#         # 获取空白图像的低频部分（低通滤波）
#         low_freq_bg = self.frequency(bg_img, filter_type='low_pass')

#         # 保存前景图像、空白图像、滤波后的高频和低频图像
#         if self.save_dir:
#             self._save_image(img, ix, "foreground")
#             self._save_image(bg_img, ix, "background")
#             self._save_image(high_freq_img, ix, "high_freq_foreground")
#             self._save_image(low_freq_bg, ix, "low_freq_background")

#         # 将当前图像的高频部分与空白图像的低频部分叠加
#         combined_img = self.combine_images(high_freq_img, low_freq_bg)

#         # 保存叠加后的新图像
#         if self.save_dir:
#             self._save_image(combined_img, ix, "combined_image")

#         return combined_img

#     def get_empty_img(self):
#         """从训练集中随机选择一个空白图像。"""
#         return self._sample_empty_img_given_mask(
#             torch.ones(len(self.empty_indices), dtype=bool)
#         )

#     def _sample_empty_img_given_mask(self, labeled_mask):
#         """根据给定的标签掩码返回一个随机的空白图像。"""
#         labeled_mask = labeled_mask.numpy()
#         if np.all(labeled_mask == 0):
#             return None

#         empty_ix = np.random.choice(self.empty_indices[labeled_mask])
#         return self.dataset.get_input(empty_ix)

#     def combine_images(self, high_freq_img, low_freq_bg):
#         """
#         将高频部分图像与低频部分图像叠加。
#         :param high_freq_img: 当前图像的高频部分
#         :param low_freq_bg: 空白图像的低频部分
#         :return: 叠加后的图像
#         """
#         # 将高频和低频部分叠加
#         high_freq_tensor = transforms.ToTensor()(high_freq_img)
#         low_freq_tensor = transforms.ToTensor()(low_freq_bg)

#         # 确保两张图像大小一致
#         high_freq_tensor = high_freq_tensor[:,:min(high_freq_tensor.shape[1], low_freq_tensor.shape[1]),:min(high_freq_tensor.shape[2], low_freq_tensor.shape[2])]
#         low_freq_tensor = low_freq_tensor[:,:min(high_freq_tensor.shape[1], low_freq_tensor.shape[1]),:min(high_freq_tensor.shape[2], low_freq_tensor.shape[2])]

#         # 叠加图像
#         combined_tensor = high_freq_tensor + low_freq_tensor
#         combined_tensor = torch.clamp(combined_tensor, 0, 1)  # 保证像素值在[0, 1]范围内

#         # 转回PIL图像
#         combined_img = transforms.ToPILImage()(combined_tensor)
#         return combined_img

#     def _save_image(self, img, ix, img_type):
#         """
#         保存图像到指定的目录。
#         :param img: 生成的图像（PIL Image对象）
#         :param ix: 图像的索引，用于生成文件名
#         :param img_type: 图像类型（'foreground'、'background'、'high_freq_foreground'、'low_freq_background'、'combined_image'）
#         """
#         if img_type not in ['foreground', 'background', 'high_freq_foreground', 'low_freq_background', 'combined_image']:
#             raise ValueError("img_type must be one of 'foreground', 'background', 'high_freq_foreground', 'low_freq_background', 'combined_image'")

#         # 创建文件名，以编号为开头
#         img_filename = f"{ix}_{img_type}.png"
#         img_path = os.path.join(self.save_dir, img_filename)
        
#         # 保存图像
#         img.save(img_path)
#         print(f"Saved {img_type} image at {img_path}")


class CopyPasteAugment:
    def __init__(self, labeled_dataset, same_y_observed=False, translate_upper=None, translate_lower=None, resize_upper=None, resize_lower=None, cutoff_frequency=1, device='cuda'):
        """
        初始化函数，用于设定图像增强的参数并创建 Frequency 类的实例。
        :param labeled_dataset: 数据集，包含图片和标签
        :param translate_upper, translate_lower: 平移增强参数
        :param resize_upper, resize_lower: 缩放增强参数
        :param cutoff_frequency: 截止频率，控制高通低通滤波器的切换频率
        :param device: 计算设备，默认为 'cuda'
        """
        assert labeled_dataset.dataset_name in ("iwildcam", "birdcalls")
        assert hasattr(labeled_dataset, 'get_bbox'), "Dataset must have bbox or mask methods"
        assert not (translate_upper is None) ^ (translate_lower is None) 
        assert not (resize_upper is None) ^ (resize_lower is None)

        self.translate_range = (translate_lower, translate_upper) if translate_upper is not None else None
        self.resize_range = (resize_lower, resize_upper) if resize_upper is not None else None
        self.dataset = labeled_dataset
        self.nargs = 2  # for the Transform class, to signal we want both img and ix passed to __call__

        self.classes_to_not_augment = [0]  # Empty class for iwildcam, birdcalls
        if 'train' in self.dataset.empty_indices:
            self.empty_indices = self.dataset.empty_indices['train']
        else:
            self.empty_indices = np.array([], dtype=int)

        # 初始化 Frequency 类
        self.frequency = Frequency(cutoff_frequency=cutoff_frequency, device=device)

    def __call__(self, img, ix):
        """
        将当前图像的高频部分与空白图像的低频部分叠加。
        """
        dataset = self.dataset

        # 不进行增强的类
        y = dataset.y_array[ix].item()
        if y in self.classes_to_not_augment: return img  # IDENTITY CASE

        # 获取相关的注释信息
        if hasattr(dataset, 'get_mask'):
            mask = dataset.get_mask(ix, resize_wh=img.size)  # 获取图像遮罩
        else:
            mask = None

        if mask is None or np.all(np.array(mask) == 0):  # 如果没有有效的遮罩
            bboxes, conf = dataset.get_bbox(ix)
            if conf < 0.5 or len(bboxes) == 0: return img  # IDENTITY CASE
            mask = create_mask_from_bboxes(bboxes, img.size)

        # 获取空白背景图像
        bg_img = self.get_empty_img()

        bg_img = bg_img.copy().resize(img.size)

        # 获取当前图像的高频部分（高通滤波）
        high_freq_img = self.frequency(img, filter_type='high_pass')

        # 获取空白图像的低频部分（低通滤波）
        low_freq_bg = self.frequency(bg_img, filter_type='low_pass')

        # 将当前图像的高频部分与空白图像的低频部分叠加
        combined_img = self.combine_images(high_freq_img, low_freq_bg)

        return combined_img

    def get_empty_img(self):
        """从训练集中随机选择一个空白图像。"""
        return self._sample_empty_img_given_mask(
            torch.ones(len(self.empty_indices), dtype=bool)
        )
    
    def get_empty_img_by_y(self, y):
        """pick an empty image from a location that contains an example with label y"""
        labeled_mask = isin(self.dataset.location_array, self.dataset.y_to_observed_locs[y])[self.empty_indices]
        return self._sample_empty_img_given_mask(labeled_mask)

    def _sample_empty_img_given_mask(self, labeled_mask):
        """根据给定的标签掩码返回一个随机的空白图像。"""
        labeled_mask = labeled_mask.numpy()
        if np.all(labeled_mask == 0):
            return None

        empty_ix = np.random.choice(self.empty_indices[labeled_mask])
        return self.dataset.get_input(empty_ix)

    def combine_images(self, high_freq_img, low_freq_bg):
        """
        将高频部分图像与低频部分图像叠加。
        :param high_freq_img: 当前图像的高频部分
        :param low_freq_bg: 空白图像的低频部分
        :return: 叠加后的图像
        """
        # 将高频和低频部分叠加
        high_freq_tensor = transforms.ToTensor()(high_freq_img)
        low_freq_tensor = transforms.ToTensor()(low_freq_bg)

        # 确保两张图像大小一致
        high_freq_tensor = high_freq_tensor[:,:min(high_freq_tensor.shape[1], low_freq_tensor.shape[1]),:min(high_freq_tensor.shape[2], low_freq_tensor.shape[2])]
        low_freq_tensor = low_freq_tensor[:,:min(high_freq_tensor.shape[1], low_freq_tensor.shape[1]),:min(high_freq_tensor.shape[2], low_freq_tensor.shape[2])]

        # 叠加图像
        combined_tensor = high_freq_tensor + low_freq_tensor
        combined_tensor = torch.clamp(combined_tensor, 0, 1)  # 保证像素值在[0, 1]范围内

        # 转回PIL图像
        combined_img = transforms.ToPILImage()(combined_tensor)
        return combined_img



# class CopyPasteAugment:
#     def __init__(self, labeled_dataset, same_y_observed=False, translate_upper=None, translate_lower=None, resize_upper=None, resize_lower=None):
#         """
#         Initialization of the augmentation. The function still retains basic functionality but now simplifies
#         the augmentation to only overlay the current image on top of an empty image.

#         """
#         assert labeled_dataset.dataset_name in ("iwildcam", "birdcalls")
#         assert hasattr(labeled_dataset, 'get_bbox'), "Dataset must have bbox or mask methods"
#         assert not (translate_upper is None) ^ (translate_lower is None) 
#         assert not (resize_upper is None) ^ (resize_lower is None)

#         self.translate_range = (translate_lower, translate_upper) if translate_upper is not None else None
#         self.resize_range = (resize_lower, resize_upper) if resize_upper is not None else None
#         self.dataset = labeled_dataset
#         self.nargs = 2  # for the Transform class, to signal we want both img and ix passed to __call__

#         self.classes_to_not_augment = [0]  # Empty class for iwildcam, birdcalls
#         if 'train' in self.dataset.empty_indices:
#             self.empty_indices = self.dataset.empty_indices['train']
#         else:
#             self.empty_indices = np.array([], dtype=int)

#     def __call__(self, img, ix):
#         """
#         Simple overlay of the current image onto a random empty image.
#         """
#         dataset = self.dataset

#         # Don't transform empty images
#         y = dataset.y_array[ix].item()
#         if y in self.classes_to_not_augment: return img  # IDENTITY CASE

#         # Get relevant annotations
#         if hasattr(dataset, 'get_mask'):
#             mask = dataset.get_mask(ix, resize_wh=img.size)  # Returns an image mask
#         else:
#             mask = None

#         if mask is None or np.all(np.array(mask) == 0):  # If the mask is empty or invalid
#             bboxes, conf = dataset.get_bbox(ix)
#             if conf < 0.5 or len(bboxes) == 0: return img  # IDENTITY CASE
#             mask = create_mask_from_bboxes(bboxes, img.size)

#         # Sample the empty background
#         bg_img = self.get_empty_img()

#         bg_img = bg_img.copy().resize(img.size)

#         # Resize if necessary (optional step)
#         if self.resize_range is not None:
#             resize_ratio = np.random.uniform(low=self.resize_range[0], high=self.resize_range[1], size=1)
#             img, mask = resize_object(img, mask, resize_ratio)

#         # Overlay current image on top of the empty background
#         bg_img.paste(img, (0, 0), mask=mask)  # Simple paste with mask
#         return bg_img

#     def get_empty_img(self):
#         """Pick an empty image from any camera location."""
#         return self._sample_empty_img_given_mask(
#             torch.ones(len(self.empty_indices), dtype=bool)
#         )

#     def _sample_empty_img_given_mask(self, labeled_mask):
#         """Return a random item from empty_indices that satisfies the given boolean masks."""
#         labeled_mask = labeled_mask.numpy()
#         if np.all(labeled_mask == 0):
#             return None

#         empty_ix = np.random.choice(self.empty_indices[labeled_mask])
#         return self.dataset.get_input(empty_ix)


# class CopyPasteAugment:
#     def __init__(self, labeled_dataset, same_cluster=False, same_y_observed=False, translate_upper=None, translate_lower=None, resize_upper=None, resize_lower=None): 
#         """
#         Assumes that the dataset has a get_bbox method that returns the bounding box of the object in the image.
#         If a get_mask method is also present, it will be used instead of the bounding box.

#         Assumes that the dataset has a empty_indices attribute that is a dict of the form {'train': [list of indices], ...},
#         where the indices are the indices of empty (no object) examples in the dataset.

#         Initialize the augmentation.
#             - same_cluster samples an empty image from the same cluster of cameras as the incoming input
#             - same_y_observed samples an empty image from a camera that in the dataset observes an instance of the same y as the incoming input
#         """
#         assert labeled_dataset.dataset_name in ("iwildcam", "birdcalls") # must have bboxes / masks
#         assert hasattr(labeled_dataset, 'get_bbox'), "You might be using a dataset class from the default wilds package. This code assumes you're using our modified version that returns bboxes, masks, etc."
#         assert not (translate_upper is None) ^ (translate_lower is None) 
#         assert not (resize_upper is None) ^ (resize_lower is None) 
#         assert np.sum([same_cluster, same_y_observed]) <= 1, "at most one of same_* flags can be on at a time"

#         self.translate_range = (translate_lower, translate_upper) if translate_upper is not None else None
#         self.resize_range = (resize_lower, resize_upper) if resize_upper is not None else None
#         self.dataset = labeled_dataset

#         self.nargs = 2 # for the Transform class, to signal we want both img and ix passed to __call__

#         self.classes_to_not_augment = [0] # 0 is the empty class for iwildcam, birdcalls

#         # prepare a list of empty images to paste onto
#         if 'train' in self.dataset.empty_indices:
#             self.empty_indices = self.dataset.empty_indices['train']
#         else:
#             self.empty_indices = np.array([], dtype=int)

#         # save other kwargs
#         if same_cluster: assert hasattr(self.dataset, 'cluster_array')
#         self.same_cluster = same_cluster
#         self.same_y_observed = same_y_observed
    
#     def __call__(self, img, ix):
#         """
#         Cut & paste the object in the img onto an empty image. Requires both the PIL image and the index of the example in the dataset.
#         """
#         dataset = self.dataset

#         # don't transform empty images
#         y = dataset.y_array[ix].item()
#         if y in self.classes_to_not_augment: return img # IDENTITY CASE

#         # get relevant annotations
#         if hasattr(dataset, 'get_mask'): mask = dataset.get_mask(ix, resize_wh=img.size) # this returns an image
#         else: mask = None
#         if mask is None or np.all(np.array(mask) == 0): # there exist cases where a mask and bboxes exist, but the mask is empty
#             bboxes, conf = dataset.get_bbox(ix)
#             if conf < 0.5 or len(bboxes) == 0: return img # IDENTITY CASE
#             mask = create_mask_from_bboxes(bboxes, img.size)
#             assert not np.all(np.array(mask) == 0), "Mask shouldn't be empty by this point."

#         # set randomly sampled translation / resizing hparams
#         if self.translate_range is not None:
#             translate_xy = np.random.uniform(low=self.translate_range[0], high=self.translate_range[1], size=2)
#             # don't move the center of the bbox off the screen
#             x_center, y_center = get_object_center(mask)
#             if x_center is not None and y_center is not None: # should be a no-op by this point; masks shouldn't be empty, but maybe the get_object_center fn fails
#                 x_center, y_center = x_center / img.width, y_center / img.height
#                 translate_xy[0], translate_xy[1] = np.clip(translate_xy[0], -x_center, 1-x_center), np.clip(translate_xy[1], -y_center, 1-y_center)
#         else:
#             translate_xy = (0, 0)
#         if self.resize_range is not None:
#             resize_ratio = np.random.uniform(low=self.resize_range[0], high=self.resize_range[1], size=1)
#         else:
#             resize_ratio = None

#         # sample the empty background
#         if self.same_cluster: 
#             bg_img = self.get_empty_img_from_cluster(dataset.cluster_array[ix])
#             if bg_img is None: return img # IDENTITY CASE
#         elif self.same_y_observed: 
#             bg_img = self.get_empty_img_by_y(y=dataset.y_array[ix].item())
#             if bg_img is None: return img # IDENTITY CASE
#         else: 
#             bg_img = self.get_empty_img()
        
#         bg_img = bg_img.copy().resize(img.size)

#         # resize if necessary
#         if resize_ratio is not None:
#             img, mask = resize_object(img, mask, resize_ratio)
        
#         # paste
#         bg_img.paste(
#             img,
#             box=(int(translate_xy[0] * bg_img.width), int(translate_xy[1] * bg_img.height)),
#             mask=mask
#         )
#         return bg_img

#     def __repr__(self) -> str:
#         format_string = self.__class__.__name__ + "("
#         for k in ('translate_range', 'resize_range'):
#             format_string += f" {k}={getattr(self, k)},"
#         for k in ('same_cluster', 'same_y_observed'):
#             if getattr(self, k): format_string += f" {k}={getattr(self, k)}"
#         format_string += ")"
#         return format_string

#     ######################################

#     def get_empty_img(self):
#         """pick an empty image from any camera location"""  
#         return self._sample_empty_img_given_mask(
#             torch.ones(len(self.empty_indices), dtype=bool)
#         )
    
#     def get_empty_img_from_cluster(self, cluster):
#         """pick an empty image from a camera cluster; some clusters have no empty images"""
#         labeled_mask = (self.dataset.cluster_array[self.empty_indices] == cluster)
#         return self._sample_empty_img_given_mask(labeled_mask)

#     def get_empty_img_by_y(self, y):
#         """pick an empty image from a location that contains an example with label y"""
#         labeled_mask = isin(self.dataset.location_array, self.dataset.y_to_observed_locs[y])[self.empty_indices]
#         return self._sample_empty_img_given_mask(labeled_mask)

#     ################################

#     def _sample_empty_img_given_mask(self, labeled_mask):
#         """return a random item from empty_indices that satisfies the given boolean masks"""
#         assert len(labeled_mask) == len(self.empty_indices)
#         # empty indices are numpy arrays, so make masks numpy

#         labeled_mask = labeled_mask.numpy()
#         if np.all(labeled_mask == 0):
#             return None

#         empty_ix = np.random.choice(self.empty_indices[labeled_mask])
#         return self.dataset.get_input(empty_ix)

# ############

# def get_object_center(mask: Image):
#     """return a point (x,y) representing the 'center' of the objects"""
#     y_objects, x_objects = np.where(np.array(mask))
#     if not (len(y_objects) or len(x_objects)): 
#         return None, None
#     else:
#         return int(np.median(x_objects)), int(np.median(y_objects))

# def resize_object(img: Image, mask, resize_ratio: float):
#     """return mask and Image resized to the resize_ratio"""
#     w, h = img.size
#     new_w, new_h = (np.array(img.size) * resize_ratio).astype(int)

#     x_object, y_object = get_object_center(mask)
#     if y_object is None or x_object is None: return img, mask
    
#     l, t, r, b = xywh_to_xyxy(x_object, y_object, w, h, xy_center=True)
#     img = img.crop((l, t, r, b)).resize((new_w, new_h)).transform(
#         (w, h), 
#         method=Image.AFFINE,
#         data=(1, 0, (new_w/2 - x_object), 0, 1, (new_h/2 - y_object))
#     ) # center object, crop, and then move back
#     mask = mask.crop((l, t, r, b)).resize((new_w, new_h)).transform(
#         (w, h), 
#         method=Image.AFFINE,
#         data=(1, 0, (new_w/2 - x_object), 0, 1, (new_h/2 - y_object))
#     )
#     return img, mask
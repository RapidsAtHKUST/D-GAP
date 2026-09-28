import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import os
import argparse
import pandas as pd
import torch
import glob
import sys
from collections import defaultdict
import multiprocessing

try:
    import wandb
except Exception as e:
    pass

# use local wilds package
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import wilds
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from wilds.common.grouper import CombinatorialGrouper

from utils import set_seed, Logger, BatchLogger, log_config, ParseKwargs, load, initialize_wandb, log_group_data, parse_bool
from train import train, evaluate
from algorithms.initializer import initialize_algorithm
from data_augmentation.transforms import initialize_transform, _parse_transform_str
from data_augmentation.batch_transform import initialize_batch_transform
from configs.utils import populate_defaults
import configs.supported as supported

import torch.multiprocessing

# Necessary for large images of GlobalWheat-WILDS
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def main():

    ''' Arg defaults are filled in according to examples/configs/ '''
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('-d', '--dataset', choices=wilds.supported_datasets, required=True)
    parser.add_argument('--algorithm', required=True, choices=supported.algorithms)
    parser.add_argument('--root_dir', required=True,
                        help='The directory where [dataset]/data can be found (or should be downloaded to, if it does not exist).')

    # Dataset
    parser.add_argument('--split_scheme', help='Identifies how the train/val/test split is constructed. Choices are dataset-specific.')
    parser.add_argument('--dataset_kwargs', nargs='*', action=ParseKwargs, default={},
                        help='keyword arguments for dataset initialization passed as key1=value1 key2=value2')
    parser.add_argument('--download', default=False, type=parse_bool, const=True, nargs='?',
                        help='If true, tries to download the dataset if it does not exist in root_dir.')
    parser.add_argument('--frac', type=float, default=1.0,
                        help='Convenience parameter that scales all dataset splits down to the specified fraction, for development purposes. Note that this also scales the test set down, so the reported numbers are not comparable with the full test set.')
    parser.add_argument('--version', default=None, type=str, help='WILDS labeled dataset version number.')

    # Loaders
    parser.add_argument('--loader_kwargs', nargs='*', action=ParseKwargs, default={})
    parser.add_argument('--train_loader', choices=['standard', 'group'])
    parser.add_argument('--uniform_over_groups', type=parse_bool, const=True, nargs='?', help='If true, sample examples such that batches are uniform over groups.')
    parser.add_argument('--distinct_groups', type=parse_bool, const=True, nargs='?', help='If true, enforce groups sampled per batch are distinct.')
    parser.add_argument('--n_groups_per_batch', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--eval_loader', choices=['standard'], default='standard')
    parser.add_argument('--gradient_accumulation_steps', type=int, help='Number of batches to process before stepping optimizer and schedulers. If > 1, we simulate having a larger effective batch size (though batchnorm behaves differently).')

    # Model
    parser.add_argument('--model', choices=supported.models)
    parser.add_argument('--model_kwargs', nargs='*', action=ParseKwargs, default={},
                        help='keyword arguments for model initialization passed as key1=value1 key2=value2')
    parser.add_argument('--pretrained_model_path', default=None, type=str, help='Specify a path to pretrained model weights')

    # Transforms
    parser.add_argument('--eval_base_transforms', default=None, type=_parse_transform_str, help='Names of transforms applied to eval examples, and never applied stochastically')
    parser.add_argument('--train_base_transforms', default=None, type=_parse_transform_str, help='Names of transforms applied to train examples, and never applied stochastically')
    parser.add_argument('--train_additional_transforms', default=None, type=_parse_transform_str, help='Names of additional transforms applied on top of base transforms to the train examples, and applied stochastically with probability transform_p')
    parser.add_argument('--transform_p', type=float, default=1.0, help='probability w/ which to apply train_additional_transforms')
    parser.add_argument('--transform_kwargs', nargs='*', action=ParseKwargs, default={}, help='transform-specific arguments, e.g. randaugment_n=2')
    parser.add_argument('--batch_transform', default=None, help='name of MixUp / CutMix style augmentation applied over batches as a unit')
    parser.add_argument('--batch_transform_kwargs', nargs='*', action=ParseKwargs, default={})
    parser.add_argument('--target_resolution', nargs='+', type=int, help='The input resolution that images will be resized to before being passed into the model. For example, use --target_resolution 224 224 for a standard ResNet.')
    parser.add_argument('--input_dropout_p', type=float)
    parser.add_argument('--to_tensor', type=bool, default=True, help="If true, converts images to tensors. If false, leaves images as PIL images.")

    # Objective
    parser.add_argument('--loss_function', choices=supported.losses)
    parser.add_argument('--loss_kwargs', nargs='*', action=ParseKwargs, default={},
                        help='keyword arguments for loss initialization passed as key1=value1 key2=value2')

    # Algorithm
    parser.add_argument('--groupby_fields', nargs='+')
    parser.add_argument('--coral_penalty_weight', type=float)
    parser.add_argument('--dann_penalty_weight', type=float)
    parser.add_argument('--dann_classifier_lr', type=float)
    parser.add_argument('--dann_featurizer_lr', type=float)
    parser.add_argument('--dann_discriminator_lr', type=float)
    parser.add_argument('--dann_class_balance_reweighting', default=False, type=parse_bool, const=True, nargs='?', help='If true, reweight the discriminator loss function to account for class imbalance.')
    parser.add_argument('--dann_use_habitats', default=False, type=parse_bool, const=True, nargs='?', help='If true, conditions DANN on Y and cluster.')
    parser.add_argument('--dann_multilinear_map', default=False, type=parse_bool, const=True, nargs='?', help='If true, conditions DANN on Y (and cluster) using a randomized multilinear map with features rather than addition.')
    parser.add_argument('--irm_lambda', type=float)
    parser.add_argument('--irm_penalty_anneal_iters', type=int)
    parser.add_argument('--algo_log_metric')
    parser.add_argument('--erm_freeze_featurizer', default=False, type=parse_bool, const=True, nargs='?', help='if true, do linear probing')

    # Model selection
    parser.add_argument('--train_split', default='train')
    parser.add_argument('--val_split', default='val')
    parser.add_argument('--val_metric')
    parser.add_argument('--val_metric_decreasing', type=parse_bool, const=True, nargs='?')

    # Optimization
    parser.add_argument('--n_epochs', type=int)
    parser.add_argument('--optimizer', choices=supported.optimizers)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--max_grad_norm', type=float)
    parser.add_argument('--optimizer_kwargs', nargs='*', action=ParseKwargs, default={},
                        help='keyword arguments for optimizer initialization passed as key1=value1 key2=value2')

    # Scheduler
    parser.add_argument('--scheduler', choices=supported.schedulers)
    parser.add_argument('--scheduler_kwargs', nargs='*', action=ParseKwargs, default={},
                        help='keyword arguments for scheduler initialization passed as key1=value1 key2=value2')
    parser.add_argument('--scheduler_metric_split', choices=['train', 'val'], default='val')
    parser.add_argument('--scheduler_metric_name')

    # Evaluation
    parser.add_argument('--process_outputs_function', choices = supported.process_outputs_functions)
    parser.add_argument('--evaluate_all_splits', type=parse_bool, const=True, nargs='?', default=True)
    parser.add_argument('--eval_splits', nargs='+', default=[])
    parser.add_argument('--eval_only', type=parse_bool, const=True, nargs='?', default=False)
    parser.add_argument('--eval_epoch', default=None, type=int, help='If eval_only is set, then eval_epoch allows you to specify evaluating at a particular epoch. By default, it evaluates the best epoch by validation performance.')

    # Misc
    parser.add_argument('--device', type=int, nargs='+', default=[0])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log_dir', default='./logs')
    parser.add_argument('--log_every', default=50, type=int)
    parser.add_argument('--save_step', type=int)
    parser.add_argument('--save_best', type=parse_bool, const=True, nargs='?', default=True)
    parser.add_argument('--save_last', type=parse_bool, const=True, nargs='?', default=True)
    parser.add_argument('--save_features', type=parse_bool, const=True, nargs='?', default=False)
    parser.add_argument('--save_pred', type=parse_bool, const=True, nargs='?', default=True)
    parser.add_argument('--no_group_logging', type=parse_bool, const=True, nargs='?')
    parser.add_argument('--progress_bar', type=parse_bool, const=True, nargs='?', default=False)
    parser.add_argument('--resume', type=parse_bool, const=True, nargs='?', default=False, help='Whether to resume from the most recent saved model in the current log_dir.')

    # Weights & Biases
    parser.add_argument('--use_wandb', type=parse_bool, const=True, nargs='?', default=False)
    parser.add_argument('--wandb_api_key_path', type=str,
                        help="Path to Weights & Biases API Key. If use_wandb is set to True and this argument is not specified, user will be prompted to authenticate.")
    parser.add_argument('--wandb_kwargs', nargs='*', action=ParseKwargs, default={},
                        help='keyword arguments for wandb.init() passed as key1=value1 key2=value2')

    config = parser.parse_args()
    config = populate_defaults(config)

    # Set device
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if len(config.device) > device_count:
            raise ValueError(f"Specified {len(config.device)} devices, but only {device_count} devices found.")

        config.use_data_parallel = len(config.device) > 1
        device_str = ",".join(map(str, config.device))
        os.environ["CUDA_VISIBLE_DEVICES"] = device_str
        config.device = torch.device("cuda")
    else:
        config.use_data_parallel = False
        config.device = torch.device("cpu")

    # Initialize logs
    if os.path.exists(config.log_dir) and config.resume:
        resume=True
        mode='a'
    elif os.path.exists(config.log_dir) and config.eval_only:
        resume=False
        mode='a'
    else:
        resume=False
        mode='w'

    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    logger = Logger(os.path.join(config.log_dir, 'log.txt'), mode)

    # Record config
    log_config(config, logger)

    # Set random seed
    set_seed(config.seed)

    # Data
    full_dataset = wilds.get_dataset(
        dataset=config.dataset,
        version=config.version,
        root_dir=config.root_dir,
        download=config.download,
        split_scheme=config.split_scheme,
        **config.dataset_kwargs)

    # Initialize grouper
    train_grouper = CombinatorialGrouper(
        dataset=full_dataset,
        groupby_fields=config.groupby_fields
    )

    # Transforms & data augmentations for labeled dataset
    # See transform.py for details
    train_transform = initialize_transform(
        base_transforms=config.train_base_transforms,
        additional_transforms=config.train_additional_transforms,
        transform_p=config.transform_p,
        config=config,
        dataset=full_dataset,
        grouper=train_grouper,
        is_training=True)
    eval_transform = initialize_transform(
        base_transforms=config.eval_base_transforms,
        config=config,
        dataset=full_dataset,
        grouper=train_grouper,
        is_training=False)

    # BatchTransforms like MixUp operate on a batch at a time
    # we'll pass this into train(), which will call this transform
    # in the for batch in the train loop
    batch_transform = initialize_batch_transform(
        batch_transform_name=config.batch_transform,
        config=config,
        dataset=full_dataset,
        batch_transform_kwargs=config.batch_transform_kwargs,
        transform_p=config.transform_p)

    # Configure labeled torch datasets (WILDS dataset splits)
    datasets = defaultdict(dict)
    for split in full_dataset.split_dict.keys():
        if split == config.train_split:
            transform = train_transform
            verbose = True
        elif split == config.val_split:
            transform = eval_transform
            verbose = True
        else:
            transform = eval_transform
            verbose = False
        # Get subset
        datasets[split]['dataset'] = full_dataset.get_subset(
            split,
            frac=config.frac,
            transform=transform)

        if split == 'train':
            datasets[split]['loader'] = get_train_loader(
                loader=config.train_loader,
                dataset=datasets[split]['dataset'],
                batch_size=256,
                uniform_over_groups=config.uniform_over_groups,
                grouper=train_grouper,
                distinct_groups=config.distinct_groups,
                n_groups_per_batch=config.n_groups_per_batch,
                **config.loader_kwargs)
        else:
            datasets[split]['loader'] = get_eval_loader(
                loader=config.eval_loader,
                dataset=datasets[split]['dataset'],
                grouper=train_grouper,
                batch_size=256,
                **config.loader_kwargs)

        # Set fields
        datasets[split]['split'] = split
        datasets[split]['name'] = full_dataset.split_names[split]
        datasets[split]['verbose'] = verbose

        # Loggers
        datasets[split]['eval_logger'] = BatchLogger(
            os.path.join(config.log_dir, f'{split}_eval.csv'), mode=mode, use_wandb=config.use_wandb
        )
        datasets[split]['algo_logger'] = BatchLogger(
            os.path.join(config.log_dir, f'{split}_algo.csv'), mode=mode, use_wandb=config.use_wandb
        )

    if config.use_wandb:
        initialize_wandb(config)

    # Logging dataset info
    # Show class breakdown if feasible
    if config.no_group_logging and full_dataset.is_classification and full_dataset.y_size==1 and full_dataset.n_classes <= 10:
        log_grouper = CombinatorialGrouper(
            dataset=full_dataset,
            groupby_fields=['y'])
    elif config.no_group_logging:
        log_grouper = None
    else:
        log_grouper = train_grouper
    log_group_data(datasets, log_grouper, logger)

    # Initialize algorithm & load pretrained weights if provided
    algorithm = initialize_algorithm(
        config=config,
        datasets=datasets,
        train_grouper=train_grouper,
    )

    if not config.eval_only:
        # Resume from most recent model in log_dir
        resume_success = False
        if resume:
            save_path = glob.glob(config.log_dir + '/*last_model.pth')
            save_path = save_path[0] if len(save_path) else None
            if save_path is None:
                epochs = [
                    int(file.split('epoch:')[1].split('_')[0])
                    for file in os.listdir(config.log_dir) if file.endswith('.pth')]
                if len(epochs) > 0:
                    latest_epoch = max(epochs)
                    save_path = glob.glob(config.log_dir + f'/*epoch:{latest_epoch}_model.pth')
                    save_path = save_path[0] if len(save_path) else None
            try:
                prev_epoch, best_val_metric = load(algorithm, save_path, device=config.device)
                epoch_offset = prev_epoch + 1
                logger.write(f'Resuming from epoch {epoch_offset} with best val metric {best_val_metric}')
                resume_success = True
            except FileNotFoundError:
                pass
        if resume_success == False:
            epoch_offset=0
            best_val_metric=None

        # Log effective batch size
        if config.gradient_accumulation_steps > 1:
            logger.write(
                (f'\nUsing gradient_accumulation_steps {config.gradient_accumulation_steps} means that')
                + (f' the effective labeled batch size is {config.batch_size * config.gradient_accumulation_steps}')
                + ('. Updates behave as if torch loaders have drop_last=False\n')
            )

        train(
            algorithm=algorithm,
            datasets=datasets,
            general_logger=logger,
            config=config,
            epoch_offset=epoch_offset,
            best_val_metric=best_val_metric,
            batch_transform=batch_transform,
            val_split=config.val_split,
            train_split=config.train_split,
        )
    else:
        if config.eval_epoch is None:
            eval_model_path = glob.glob(config.log_dir + '/*best_model.pth')
            eval_model_path = eval_model_path[0] if len(eval_model_path) else None
        else:
            eval_model_path = glob.glob(config.log_dir + f'/*epoch:{config.eval_epoch}_model.pth')
            eval_model_path = eval_model_path[0] if len(eval_model_path) else None
        best_epoch, best_val_metric = load(algorithm, eval_model_path, device=config.device)
        if config.eval_epoch is None:
            epoch = best_epoch
        else:
            epoch = config.eval_epoch
        if epoch == best_epoch:
            is_best = True
        evaluate(
            algorithm=algorithm,
            datasets=datasets,
            epoch=epoch,
            general_logger=logger,
            config=config,
            is_best=is_best)

    if config.use_wandb:
        wandb.finish()
    logger.close()
    for split in datasets:
        datasets[split]['eval_logger'].close()
        datasets[split]['algo_logger'].close()

if __name__=='__main__':
    multiprocessing.set_start_method('spawn', force=True)
    main()






import numpy as np
import torch
# from wilds.common.utils import MixedY
from data_augmentation.utils import sample_rectangle
import os

IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
IMNET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

def initialize_batch_transform(
    batch_transform_name, config, dataset, batch_transform_kwargs, transform_p=1.0,
):
    if batch_transform_name is None:
        return None

    assert dataset.is_classification
    assert config.algorithm == 'ERM'

    if batch_transform_name == 'mixup':
        augmentation = MixUp(p=transform_p, alpha =0.5)
    elif batch_transform_name == 'cutmix':
        augmentation = CutMix(p=transform_p, **batch_transform_kwargs)
    elif batch_transform_name == 'freqmix':
        augmentation = DGAPAugment(labeled_dataset=dataset, p=transform_p)
    else:
        raise ValueError(f"{batch_transform_name} not recognized")
    return augmentation

#################

class BatchTransform:
    """
    These augmentations operate on batches of examples provided by a torch dataloader, rather than 
    individual examples. Though we initialize a BatchTransform object early on and pass it into train(),
    the transform isn't actually called until train() and we start iterating through the data loader.
    """
    def __init__(self, p, allow_grad: bool = False):
        self.p = p # probability of applying the augmentation to a batch
        self.allow_grad = allow_grad

    
    def __call__(self, x, y, m):
        assert torch.is_tensor(x) and torch.is_tensor(y) and torch.is_tensor(m)
        assert y.ndim == 1  # simple prediction only

        random_apply =  (self.p >= torch.rand(1)).item()
        if (random_apply == True): # with some p, transform the batch
            # with torch.no_grad():
            #     return self._transform_batch(x, y, m)
            if self.allow_grad:
                # Transformations that require gradients (e.g., gradient-guided frequency mixing)
                return self._transform_batch(x, y, m)
            else:
                # Pure data augmentation; disable gradients for efficiency
                with torch.no_grad():
                    return self._transform_batch(x, y, m)
        else:
            return x, y, m # do nothing
        
    def _transform_batch(self, x, y, m):
        raise NotImplementedError

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for k, v in self.__dict__.items(): 
            format_string += f" {k}={v},"
        format_string += ")"
        return format_string

class MixUp(BatchTransform):
    def __init__(self, p, alpha, within_y=False):
        assert alpha > 0
        self.alpha = alpha
        self.within_y = within_y
        super().__init__(p)

    def _transform_batch(self, x, y, m):
        """
        Assume:
            - x is a tensor of shape (B, ...) where B is the batch size
            - y is a tensor of shape (B, )
            - m is a tensor of shape (B, ...)
        """
        batch_size = x.shape[0]
        lmbda = np.random.beta(self.alpha, self.alpha)

        if self.within_y:
            # for each x, select another x' that has the same y, and include the option of selecting x itself
            x1, x2 = x, x[[np.random.choice(np.where(y == yp)[0]) for yp in y]]
            y1, y2 = y, y
        else:
            rand_index = torch.randperm(batch_size) # on cpu
            y1, y2 = y, y[rand_index]
            x1, x2 = x, x[rand_index]

        x = lmbda * x1 + (1-lmbda) * x2
        # y = MixedY(y1, y2, lmbda)
        m = m # don't mix metadata -- shouldn't affect ERM
        return x, y, m

class CutMix(BatchTransform):
    def __init__(self, p, alpha, within_y=False):
        assert alpha > 0
        self.alpha = alpha
        self.within_y = within_y
        super().__init__(p)

    def _transform_batch(self, x, y, m):
        """
        Assume:
            - x is a tensor of shape (B, ...) where B is the batch size
            - y is a tensor of shape (B, )
            - m is a tensor of shape (B, ...)
        """
        batch_size, _, w, h = x.shape
        lmbda = np.random.beta(self.alpha, self.alpha)
        
        bbox = sample_rectangle(w, h, w*np.sqrt(1-lmbda), h*np.sqrt(1-lmbda))
        bbox_l, bbox_t = int(bbox[0]), int(bbox[2])
        bbox_r, bbox_b = int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])

        if self.within_y:
            # for each x, select another x' that has the same y, and include the option of selecting x itself
            rand_index = [np.random.choice(np.where(y == yp)[0]) for yp in y]
        else:
            rand_index = torch.randperm(batch_size) # on cpu
        
        x[:, :, bbox_l:bbox_r, bbox_t:bbox_b] = x[rand_index, :, bbox_l:bbox_r, bbox_t:bbox_b] 
        lmbda = 1 - ((bbox_r - bbox_l) * (bbox_b - bbox_t)) / (w * h) # adjust to be the pixel ratio
        y1, y2 = y, y[rand_index]
        # y = MixedY(y1, y2, lmbda)            
        m = m # don't mix metadata -- shouldn't affect ERM
        return x, y, m



import os
import math
import random
from typing import Optional, Tuple, List

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from data_augmentation.utils import create_mask_from_bboxes  
from data_augmentation.frequency import *  


class DGAPAugment(BatchTransform):
    """
    - Input/output are (x, y, m), where x: [B, C, H, W], y: [B], m: arbitrary metadata.
    - For each sample in the batch, we draw a background image from a different hospital/site
      (if metadata is available) and perform low-frequency amplitude mixing in the Fourier domain.
      If a model is injected, we can additionally use gradient-guided weights d for mixing;
      otherwise we fall back to a random-λ low-frequency mixing.
    - y and m remain unchanged (the augmentation does not alter labels or metadata).
    """

    def __init__(
        self,
        labeled_dataset,
        p: float = 1.0,                      # Trigger probability, following the BatchTransform convention
        ratio: float = 1.0,                  # Area ratio of the low-frequency window (edge length scales with sqrt(ratio))
        cutoff_frequency: int = 1,
        device: str = "cuda",
        save_dir: Optional[str] = None,
        per_channel: bool = False,           # Whether to normalize gradient-based weights per channel
        clamp_minmax: Tuple[float,float] = (0.0, 1.0),  # Clamp range after inverse FFT
    ):
        self.dataset = labeled_dataset
        self.cutoff_frequency = cutoff_frequency
        self.device = device
        self.save_dir = save_dir
        self.ratio = float(ratio)
        self.per_channel = per_channel
        self.clamp_minmax = clamp_minmax

        self.model: Optional[torch.nn.Module] = None

        # Model will be attached later via set_model / bind_model_to_DGAP
        self._has_metadata = hasattr(self.dataset, "_metadata_array")
        self._has_y = hasattr(self.dataset, "_y_array")

        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)

        super().__init__(p=p, allow_grad=True)  # 交给 BatchTransform 处理触发概率

    # ---------------------------
    # Model injection & utilities
    # ---------------------------
    def set_model(self, model: torch.nn.Module):
        """
        Attach a torch.nn.Module to be used for gradient-guided amplitude mixing.
        If a DataParallel/DistributedDataParallel wrapper is passed, unwrap the .module.
        """
        if hasattr(model, "module") and isinstance(model.module, torch.nn.Module):
            model = model.module
        if not isinstance(model, torch.nn.Module):
            raise ValueError("set_model expects a torch.nn.Module")
        self.model = model

    @staticmethod
    def _imnet_norm(x: torch.Tensor) -> torch.Tensor:
        """ImageNet mean/std normalization. Input: [B,3,H,W], assumed in [0,1]."""
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        return (x - mean) / std

    def _predict_label_from_img_tensor(self, x01: torch.Tensor) -> int:
        if self.model is None:
            raise RuntimeError("Model has not been set; cannot predict label.")
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self._imnet_norm(x01))
            return int(torch.argmax(logits, dim=1).item())

    def _get_image_by_index(self, idx: int) -> Image.Image:
        """
        Retrieve a (preferably) untransformed image from the dataset by index,
        and convert it into a PIL.Image if needed.
        """
        if hasattr(self.dataset, "get_input"):
            x = self.dataset.get_input(idx)
        else:
            item = self.dataset[idx]
            x = item[0] if isinstance(item, (list, tuple)) and len(item) >= 1 else item

        if isinstance(x, Image.Image):
            return x

        if torch.is_tensor(x):
            x = x.detach().cpu()
            if x.ndim == 3:
                return TF.to_pil_image(x.clamp(0, 1))
            elif x.ndim == 4:
                return TF.to_pil_image(x[0].clamp(0, 1))

        if isinstance(x, np.ndarray):
            if x.ndim == 3:
                return Image.fromarray(x)
            elif x.ndim == 4:
                return Image.fromarray(x[0])

        raise RuntimeError(f"Failed to obtain image from dataset at index {idx}")

    def _sample_bg_from_other_hosp(self, target_hosp: Optional[int], H: int, W: int) -> Image.Image:
        """
        Sample a background image from a different hospital/center.
        If metadata is not available or no different hospital exists, sample globally.
        """
        n = len(self.dataset)
        if n <= 1:
            return Image.new("RGB", (W, H), color=0)

        if self._has_metadata and target_hosp is not None:
            # Try multiple times to sample a different hospital
            for _ in range(30):
                j = random.randrange(n)
                hosp = int(self.dataset._metadata_array[j][0])
                if hosp != target_hosp:
                    return self._get_image_by_index(j).copy().resize((W, H))
        # Fallback: random index without hospital constraint
        j = random.randrange(n)
        return self._get_image_by_index(j).copy().resize((W, H))

    # ---------------------------
    # BatchTransform interface
    # ---------------------------
    def _transform_batch(self, x: torch.Tensor, y: torch.Tensor, m: torch.Tensor):
        """
        Apply cross-domain, low-frequency amplitude mixing to each sample in the batch.
        x: [B, C, H, W], y: [B], m: arbitrary metadata.
        Returns: (x_mixed, y, m)
        """
        assert torch.is_tensor(x) and torch.is_tensor(y) and torch.is_tensor(m)

        B, C, H, W = x.shape
        device = x.device

        out_list: List[torch.Tensor] = []

        mean = IMNET_MEAN.to(device)
        std  = IMNET_STD.to(device)
        x1 = (x * std + mean)

        for i in range(B):
            xi = x1[i].detach().clone()
            # Assume xi is approximately in [0,1]; if not, an extra rescaling could be applied here.
            xi01 = xi.clamp(0.0, 1.0)
            pil_xi = TF.to_pil_image(xi01.cpu())

            target_hosp = None
            if self._has_metadata:
                try:
                    target_hosp = int(m[i][0].item()) if torch.is_tensor(m) else None
                except Exception:
                    target_hosp = None

            bg_img = self._sample_bg_from_other_hosp(target_hosp, H, W)

            # Use the ground-truth label (if available) for potential gradient-guided mixing
            label_i = int(y[i].item()) if self._has_y else None

            if self.model is not None:
                mixed_pil = self.colorful_spectrum_mix(
                    img1=pil_xi,
                    img2=bg_img,
                    model=self.model,
                    label=label_i,
                    ratio=self.ratio,
                    per_channel=self.per_channel,
                )
            else:
                mixed_pil, _ = self.colorful_spectrum_mix_cuda1(
                    img1=pil_xi,
                    img2=bg_img,
                    alpha=0.9,
                    ratio=self.ratio,
                )

            mixed_t = TF.to_tensor(mixed_pil).to(device)
            # Keep channel count and clamp to the specified range
            mixed_t = mixed_t.clamp(*self.clamp_minmax)
            out_list.append(mixed_t)


        x_mixed = torch.stack(out_list, dim=0)  # [B,C,H,W]
        # Re-apply ImageNet normalization (broadcast to batch and spatial dimensions)
        mean = IMNET_MEAN.to(x_mixed.device, x_mixed.dtype)
        std  = IMNET_STD.to(x_mixed.device, x_mixed.dtype)
        x_mixed = (x_mixed - mean) / std  
        return x_mixed, y, m


    def colorful_spectrum_mix_cuda1(
        self, img1: Image.Image, img2: Image.Image, alpha: float = 0.9, ratio: Optional[float] = None
    ) -> Tuple[Image.Image, float]:
        if ratio is None:
            ratio = self.ratio
        lam = random.random() * 0.7

        x1 = TF.to_tensor(img1).unsqueeze(0)
        x2 = TF.to_tensor(img2).unsqueeze(0)
        _, _, H, W = x1.shape

        X1 = torch.fft.fft2(x1, dim=(-2, -1))
        A1, P1 = torch.abs(X1), torch.angle(X1)
        X2 = torch.fft.fft2(x2, dim=(-2, -1))
        A2 = torch.abs(X2)

        h0, w0 = H // 2, W // 2
        h_crop = int(H * math.sqrt(ratio));  w_crop = int(W * math.sqrt(ratio))
        h1, h2 = h0 - h_crop // 2, h0 + h_crop // 2
        w1, w2 = w0 - w_crop // 2, w0 + w_crop // 2

        A_mix = A1.clone()
        A_mix[:, :, h1:h2, w1:w2] = lam * A2[:, :, h1:h2, w1:w2] + (1 - lam) * A1[:, :, h1:h2, w1:w2]
        Y = torch.fft.ifft2(A_mix * torch.exp(1j * P1), dim=(-2, -1)).real.clamp(0, 1)
        out = (Y * 255.0).round().byte().squeeze(0).permute(1, 2, 0).cpu().numpy()


        return Image.fromarray(out), lam



    def colorful_spectrum_mix(
        self,
        img1: Image.Image,
        img2: Image.Image,
        model: Optional[torch.nn.Module] = None,
        label: Optional[int] = None,
        ratio: float = 1.0,
        per_channel: bool = False,
        eps: float = 1e-6,
    ) -> Image.Image:
        """
        Low-frequency rectangular-window amplitude fusion:
        A_mix = (1 - d) * A1 + d * A2, keeping the phase of img1.
        If model is None, this falls back to a random-λ mixing.
        """
        x1 = TF.to_tensor(img1).unsqueeze(0)
        x2 = TF.to_tensor(img2).unsqueeze(0)
        _, C, H, W = x1.shape

        if model is None:
            # Random-λ low-frequency fusion
            lam = torch.rand(1).item() * 0.7
            X1 = torch.fft.fft2(x1, dim=(-2, -1))
            A1, P1 = torch.abs(X1), torch.angle(X1)
            X2 = torch.fft.fft2(x2, dim=(-2, -1))
            A2 = torch.abs(X2)
            h0, w0 = H // 2, W // 2
            h_crop = int(H * math.sqrt(ratio));  w_crop = int(W * math.sqrt(ratio))
            h1, h2 = h0 - h_crop // 2, h0 + h_crop // 2
            w1, w2 = w0 - w_crop // 2, w0 + w_crop // 2
            A1m = A1.clone()
            A1m[:, :, h1:h2, w1:w2] = lam * A2[:, :, h1:h2, w1:w2] + (1 - lam) * A1[:, :, h1:h2, w1:w2]
            Y = torch.fft.ifft2(A1m * torch.exp(1j * P1), dim=(-2, -1)).real.clamp(0, 1)
            out = (Y * 255.0).round().byte().squeeze(0).permute(1, 2, 0).cpu().numpy()
            return Image.fromarray(out)

        device = next(model.parameters()).device
        x1 = x1.to(device); x2 = x2.to(device)

        # Frequency decomposition
        X1 = torch.fft.fft2(x1, dim=(-2, -1))
        A1, P1 = torch.abs(X1), torch.angle(X1)
        X2 = torch.fft.fft2(x2, dim=(-2, -1))
        A2 = torch.abs(X2)

        # Low-frequency window
        h0, w0 = H // 2, W // 2
        h_crop = int(H * math.sqrt(ratio));  w_crop = int(W * math.sqrt(ratio))
        h1, h2 = h0 - h_crop // 2, h0 + h_crop // 2
        w1, w2 = w0 - w_crop // 2, w0 + w_crop // 2

        # Temporarily disable BN updates
        was_training = model.training
        model.eval()

        # Compute gradient w.r.t. A1 to construct weight d
        A1_req = A1.detach().clone().requires_grad_(True)
        Y1 = torch.fft.ifft2(A1_req * torch.exp(1j * P1), dim=(-2, -1)).real.clamp(0, 1)

        if label is None:
            y = torch.tensor([self._predict_label_from_img_tensor(Y1)], device=device)
        else:
            y = torch.tensor([int(label)], device=device)

        logits = model(self._imnet_norm(Y1))
        ce = F.cross_entropy(logits, y)
        g = torch.autograd.grad(ce, A1_req, retain_graph=False, create_graph=False)[0].abs()

        # Restore training mode
        model.train(was_training)

        # Normalize gradient in the low-frequency window to obtain d in [0,1]
        gw = g[:, :, h1:h2, w1:w2]
        if per_channel:
            gf = gw.reshape(1, C, -1)
            gmax = gf.max(-1, keepdim=True).values.view(1, C, 1, 1)
            gmin = gf.min(-1, keepdim=True).values.view(1, C, 1, 1)
            d = (gw - gmin) / (gmax - gmin + 1e-6)
        else:
            gmean = gw.mean(1, keepdim=True)
            gf = gmean.reshape(1, -1)
            gmax = gf.max(-1, keepdim=True).values.view(1, 1, 1, 1)
            gmin = gf.min(-1, keepdim=True).values.view(1, 1, 1, 1)
            d = (gmean - gmin) / (gmax - gmin + 1e-6)
            d = d.expand(-1, C, -1, -1)
            d_flat = d.reshape(-1)
            mean = d_flat.mean()
            std = d_flat.std() + 1e-6
            d_norm = (d - mean) / std
            d = torch.sigmoid(d_norm)
            d = d.clamp(0.4, 0.7)

        # Low-frequency fusion with gradient-guided weights
        A_mix = A1.clone()
        A_mix[:, :, h1:h2, w1:w2] = (1.0 - d) * A1[:, :, h1:h2, w1:w2] + d * A2[:, :, h1:h2, w1:w2]

        # Inverse FFT back to image space
        Y = torch.fft.ifft2(A_mix * torch.exp(1j * P1), dim=(-2, -1)).real.clamp(0, 1)
        out = (Y * 255.0).round().byte().squeeze(0).permute(1, 2, 0).cpu().numpy()
        

        return Image.fromarray(out)



# ---------------------------
# bind a model to all DGAPAugment instances
# ---------------------------
def _extract_torch_model(maybe_algo):
    import torch as _torch
    for name in ["model", "network", "classifier", "module"]:
        if hasattr(maybe_algo, name) and isinstance(getattr(maybe_algo, name), _torch.nn.Module):
            return getattr(maybe_algo, name)
    if isinstance(maybe_algo, _torch.nn.Module):
        return maybe_algo
    return None

def bind_model_to_DGAP(root_transform, model):
    try:
        if hasattr(root_transform, "transforms") and isinstance(root_transform.transforms, list):
            for t in root_transform.transforms:
                bind_model_to_DGAP(t, model)
        if hasattr(root_transform, "branches") and isinstance(root_transform.branches, list):
            for b in root_transform.branches:
                bind_model_to_DGAP(b, model)
        if hasattr(root_transform, "transform"):
            bind_model_to_DGAP(root_transform.transform, model)
        if isinstance(root_transform, DGAPAugment):
            root_transform.set_model(model)
    except Exception:
        pass

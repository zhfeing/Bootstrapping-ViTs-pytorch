import logging
import copy
from typing import Dict, Any

import torch.utils.data as data

import cv_lib.classification.data as cls_data
import cv_lib.distributed.utils as dist_utils
from cv_lib.distributed.sampler import get_train_sampler, get_val_sampler

from .aug import get_data_aug
from vit_mutual.utils import DistLaunchArgs


def build_eval_dataset(
    data_cfg: Dict[str, Any],
    val_cfg: Dict[str, Any],
    launch_args: DistLaunchArgs,
):
    logger = logging.getLogger("build_eval_dataset")
    # get dataloader
    data_cfg = copy.deepcopy(data_cfg)
    name = data_cfg.pop("name")
    dataset = cls_data.__REGISTERED_DATASETS__[name]
    root = data_cfg.pop("root")
    val_data_cfg = data_cfg.pop("val")
    val_aug = get_data_aug(name, "val")
    data_cfg.pop("train", None)

    val_dataset: cls_data.ClassificationDataset = dataset(
        root=root,
        augmentations=val_aug,
        **val_data_cfg,
        **data_cfg
    )

    n_classes = val_dataset.n_classes
    if dist_utils.is_main_process():
        logger.info(
            "Loaded %s dataset with %d val examples, %d classes",
            name, len(val_dataset), n_classes
        )
    dist_utils.barrier()
    val_sampler = get_val_sampler(launch_args.distributed, val_dataset)
    val_bs = val_cfg["batch_size"]
    val_workers = val_cfg["num_workers"]
    if launch_args.distributed:
        val_bs, val_workers = dist_utils.cal_split_args(
            val_bs,
            val_workers,
            launch_args.ngpus_per_node
        )
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=val_bs,
        num_workers=val_workers,
        pin_memory=True,
        sampler=val_sampler
    )
    logger.info(
        "Build validation dataset done\nEval: %d imgs, %d batchs",
        len(val_dataset),
        len(val_loader)
    )
    return val_loader, n_classes


def build_train_dataset(
    data_cfg: Dict[str, Any],
    train_cfg: Dict[str, Any],
    val_cfg: Dict[str, Any],
    launch_args: DistLaunchArgs,
):
    logger = logging.getLogger("build_train_dataset")
    # get dataloader
    train_aug = get_data_aug(data_cfg["name"], "train")
    val_aug = get_data_aug(data_cfg["name"], "val")
    train_dataset, val_dataset, n_classes = cls_data.get_dataset(
        data_cfg,
        train_aug,
        val_aug
    )
    if dist_utils.is_main_process():
        logger.info(
            "Loaded %s dataset with %d train examples, %d val examples, %d classes",
            data_cfg["name"], len(train_dataset), len(val_dataset), n_classes
        )
    dist_utils.barrier()
    train_sampler = get_train_sampler(launch_args.distributed, train_dataset)
    val_sampler = get_val_sampler(launch_args.distributed, val_dataset)
    train_bs = train_cfg["batch_size"]
    train_workers = train_cfg["num_workers"]
    val_bs = val_cfg["batch_size"]
    val_workers = val_cfg["num_workers"]
    if launch_args.distributed:
        train_bs, train_workers = dist_utils.cal_split_args(
            train_bs,
            train_workers,
            launch_args.ngpus_per_node
        )
        val_bs, val_workers = dist_utils.cal_split_args(
            val_bs,
            val_workers,
            launch_args.ngpus_per_node
        )
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=train_bs,
        num_workers=train_workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True
    )
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=val_bs,
        num_workers=val_workers,
        pin_memory=True,
        sampler=val_sampler
    )
    logger.info(
        "Build train dataset done\nTraining: %d imgs, %d batchs\nEval: %d imgs, %d batchs",
        len(train_dataset),
        len(train_loader),
        len(val_dataset),
        len(val_loader)
    )
    return train_loader, val_loader, n_classes


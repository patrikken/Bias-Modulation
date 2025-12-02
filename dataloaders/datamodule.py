from lightning import LightningDataModule
from lightning.pytorch.utilities import CombinedLoader
from dataloaders.dataset import BaseDataset, SubSet
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import math
import torch
from torch.utils.data import Sampler
from collections import defaultdict
import random
import torch.distributed as dist
import os
import logging
import numpy as np

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12355"


def init_distributed_mode(dist_url="env://", timeout=1800):
    if dist_url == "env://" and "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        global_rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        logging.info(
            f"Distributed init (env://): global_rank: {global_rank}, world_size: {world_size}"
        )
    else:
        global_rank = 1
        world_size = 1
        logging.info("Distributed not initialized, running with a single process.")
        # return
    if torch.cuda.is_available():
        torch.cuda.set_device(global_rank % torch.cuda.device_count())
    backend = "nccl"
    dist.init_process_group(
        backend=backend,
        init_method=dist_url,
        world_size=world_size,
        rank=global_rank,
        # timeout=timeout,
    )


class BaseDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        model_class: BaseDataset,
        pin_memory=True,
        persistent_workers=True,
        sensitive_attr: list = None,
    ):
        super().__init__()
        self.model_class = model_class
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.dataset: BaseDataset = None

        self.group_id = None  # Group id for loading data of a specific group mainly used for the Decoupled model.

    def get_model_class_params(self):
        raise NotImplementedError

    def prepare_data(self):
        # Implement data downloading or preprocessing here
        self.model_class(**self.get_model_class_params()).download()
        pass

    def setup(self, stage=None):
        # Implement data splitting and transformations here

        if stage == "fit" or stage is None:
            self.dataset = self.model_class(
                **self.get_model_class_params(), train=True
            ).preprocess()
            if self.dataset.val_idx is None:
                (
                    self.dataset.train_idx,
                    self.dataset.val_idx,
                ) = train_test_split(
                    self.dataset.train_idx,
                    test_size=0.1,
                    random_state=42,
                    stratify=self.dataset.target[self.dataset.train_idx],
                )
            self.train = SubSet(self.dataset, self.dataset.train_idx)
            self.val = SubSet(self.dataset, self.dataset.val_idx)
            self.input_dim = self.dataset.input_dim
        elif stage == "test":
            self.dataset = self.model_class(
                **self.get_model_class_params()
            ).preprocess()
            self.test = SubSet(self.dataset, self.dataset.test_idx)
            self.input_dim = self.dataset.input_dim

    def set_current_group(self, group_id):
        self.group_id = group_id

    def train_dataloader(self):
        # Return the training dataloader
        if self.dataset.is_decoupled:
            if self.group_id is None:
                raise ValueError("Must call `set_group_dataloader` before")
            group_indices = np.where(
                self.dataset.group_idx[self.dataset.train_idx] == self.group_id
            )[0]
            return self._loader(group_indices, True)
        return self._loader(self.dataset.train_idx, shuffle=True)

    def val_dataloader(self):
        # Return the validation dataloader
        if self.dataset.val_idx is not None:
            if self.dataset.is_decoupled:
                if self.group_id is None:
                    raise ValueError("Must call `set_group_dataloader` before")
                group_indices = np.where(
                    self.dataset.group_idx[self.dataset.val_idx] == self.group_id
                )[0]
                return self._loader(group_indices, False)
            return self._loader(self.dataset.val_idx, shuffle=False)
        else:
            return None

    def test_dataloader(self):
        # Return the test dataloader
        return self._loader(self.dataset.test_idx, shuffle=False)

    def _loader(self, indices, shuffle=True):

        data = SubSet(self.dataset, indices)

        """ sampler = GroupedBatchSampler(
            group_ids=data.group_idx,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=True,
        ) """
        return DataLoader(
            data,
            self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            # sampler=sampler,
        )


class GroupedBatchSampler(Sampler):
    """
    Sampler that groups samples by group_id and ensures each batch
    contains samples from only one group.

    group_ids : array-like, group id per sample
    batch_size : samples per batch
    shuffle : shuffle groups and within-group indices each epoch
    drop_last : drop last incomplete batch
    """

    def __init__(
        self,
        group_ids,
        batch_size,
        shuffle=True,
        drop_last=False,
    ):
        self.group_ids = list(group_ids)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.epoch = 0

        # Build mapping group_id → list of indices
        self.group_to_indices = defaultdict(list)
        for i, gid in enumerate(self.group_ids):
            self.group_to_indices[gid].append(i)

        self.groups = list(self.group_to_indices.keys())

    def __iter__(self):
        g = self._generate_epoch_batches()
        return iter(g)

    def __len__(self):
        batches = self._generate_epoch_batches(count_only=True)
        return batches

    def set_epoch(self, epoch):
        """Must be called by training loop."""
        self.epoch = epoch

    # ------------------------------
    # Internal helper
    # ------------------------------
    def _generate_epoch_batches(self, count_only=False):
        # 1. Shuffle groups per epoch
        groups = self.groups.copy()
        if self.shuffle:
            g_rng = random.Random(self.epoch)
            g_rng.shuffle(groups)

        total_batches = 0
        batches = []

        for gid in groups:
            idxs = self.group_to_indices[gid]

            # Shuffle inside-group
            if self.shuffle:
                rng = random.Random(self.epoch + hash(gid))
                idxs = idxs.copy()
                rng.shuffle(idxs)

            # Form batches for this group
            for i in range(0, len(idxs), self.batch_size):
                batch = idxs[i : i + self.batch_size]

                if len(batch) < self.batch_size and self.drop_last:
                    continue

                if count_only:
                    total_batches += 1
                else:
                    batches.append(batch)

        if count_only:
            return total_batches
        return batches


class DistributedGroupedBatchSampler(Sampler):
    """
    Distributed sampler that groups samples by group_id and ensures each batch
    contains samples from only one group. Each GPU gets its own shard.

    group_ids : array-like, group id per sample
    batch_size : samples per batch
    num_replicas : number of GPUs (auto-detected if None)
    rank : current GPU rank (auto-detected if None)
    shuffle : shuffle groups and within-group indices each epoch
    drop_last : drop last incomplete batch on rank
    """

    def __init__(
        self,
        group_ids,
        batch_size,
        num_replicas=None,
        rank=None,
        shuffle=True,
        drop_last=False,
    ):
        init_distributed_mode()
        if num_replicas is None:
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            rank = torch.distributed.get_rank()

        self.group_ids = list(group_ids)
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.epoch = 0

        # Build mapping group_id → list of indices
        self.group_to_indices = defaultdict(list)
        for i, gid in enumerate(self.group_ids):
            self.group_to_indices[gid].append(i)

        self.groups = list(self.group_to_indices.keys())

    def __iter__(self):
        g = self._generate_epoch_batches()
        return iter(g)

    def __len__(self):
        batches = self._generate_epoch_batches(count_only=True)
        return batches

    def set_epoch(self, epoch):
        """Must be called by DistributedDataParallel training loop."""
        self.epoch = epoch

    # ------------------------------
    # Internal helper
    # ------------------------------
    def _generate_epoch_batches(self, count_only=False):
        # 1. Shuffle groups per epoch
        groups = self.groups.copy()
        if self.shuffle:
            g_rng = random.Random(self.epoch)
            g_rng.shuffle(groups)

        total_local_batches = 0
        batches = []

        for gid in groups:
            idxs = self.group_to_indices[gid]

            # Shuffle inside-group
            if self.shuffle:
                rng = random.Random(self.epoch + hash(gid))
                idxs = idxs.copy()
                rng.shuffle(idxs)

            # Split inside-group indices *per rank* (shard them)
            # Each rank gets every n-th shard-like segment
            shard = idxs[self.rank :: self.num_replicas]

            # Form batches for this rank
            for i in range(0, len(shard), self.batch_size):
                batch = shard[i : i + self.batch_size]

                if len(batch) < self.batch_size and self.drop_last:
                    continue

                if count_only:
                    total_local_batches += 1
                else:
                    batches.append(batch)

        if count_only:
            return total_local_batches
        return batches

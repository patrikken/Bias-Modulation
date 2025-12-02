import argparse
import os
import random
import sys
import numpy as np
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch import Trainer

from models.mlp import MLP
import torch
import pandas as pd

from dataloaders.heart import HeartDataset
from dataloaders.datamodule import BaseDataModule
from utils.seed_all import seed_everything


if __name__ == "__main__":
    h = HeartDataset(train=True)
    h.download()
    print(h.data.shape, h.data[0:4, :])

import pandas as pd
import os
import os
import random
import numpy as np
import torch
import logging


def save_test_results(result_dir, results, seed=1):
    logging.info("Saving results in %s ", result_dir)
    os.makedirs(result_dir, exist_ok=True)
    # Save test results
    result_path = os.path.join(result_dir, f"seed_{seed}.csv")
    df = pd.DataFrame(results)
    df.to_csv(result_path, index=False)


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

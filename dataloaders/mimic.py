from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from dataloaders.datamodule import BaseDataModule
from dataloaders.dataset import BaseDataset
import os
import numpy as np
import logging


class MimicDataset(BaseDataset):
    def __init__(
        self,
        sensitive_attr=["age", "sex"],
        train=False,
        data_dir="./data/mortality",
        is_1hot=False,
        is_intersectional=False,
        is_decoupled=False,
    ):
        super().__init__(sensitive_attr, train, is_1hot, is_intersectional)

        # assert (
        #    is_1hot + is_intersectional + is_decoupled >= 1
        # ), "Only one of is_1hot, is_intersectional, is_decoupled can be True. They must be mutually exclusive."

        self.sensitive_attr = sensitive_attr
        self.data_dir = data_dir
        self.name = f"{'_'.join(sensitive_attr)}"
        self.group_idx = None  # Group IDs for each sample, to be set in preprocess. Example of values: {'00'=>0, '01'=>1, '10'=>2, '11'=>3} for two binary sensitive attributes

        self.is_intersectional = is_intersectional
        self.is_1hot = is_1hot
        self.is_decoupled = is_decoupled

    def __len__(self):
        return len(self.data)

    def process_data_common(self, data):
        def _get_group_label(row, columns):
            toRet = ""
            for c in columns:
                # toRet += str(row[c])
                toRet += f"{c}[{int(row[c])}]"
            return toRet

        self.group_label = data[self.sensitive_attr]  # Example sensitive attribute
        self.group_idx = self.group_label.apply(
            lambda row: _get_group_label(row, self.sensitive_attr), axis=1
        )
        le = LabelEncoder()
        self.group_idx = le.fit_transform(
            self.group_idx
        )  # Convert group labels to numerical indices
        self.group_label_names = le.classes_.tolist()
        logging.info(
            f"Group labels: {self.group_label_names}. Showing list of unique group labels encoded with label encoding."
        )
        self.data = data.drop(
            columns=self.sensitive_attr
        )  # Remove sensitive attribute from features

        if self.is_1hot:
            group_1hot = pd.get_dummies(self.group_label, drop_first=False)
            self.data = pd.concat([self.data, group_1hot], axis=1)
        elif self.is_intersectional:
            # One-hot encode intersectional groups
            group_1hot = pd.get_dummies(
                self.group_idx, drop_first=False, prefix="group"
            )

            self.data = pd.concat(
                [self.data, group_1hot], axis=1, ignore_index=True
            )  # Append intersectional group one-hot encodings to data

        column_list = self.data.columns.tolist()
        scaler = StandardScaler()
        self.data = scaler.fit_transform(self.data).astype("float32")

        self.train_idx, self.test_idx = train_test_split(
            range(len(self.data)),
            test_size=0.2,
            stratify=self.target,
        )
        """ print(
            f"Loaded Heart\n List of features: {column_list} \n {'Train' if self.train else 'Test'} \n dataset size: {len(self.train_idx) if self.train else len(self.test_idx)} samples"
        ) """
        self.input_dim = self.data.shape[1]
        self.group_label = self.group_label.values
        self.target = self.target.values
        self.nb_groups = len(self.sensitive_attr)  # number of sensitive attributes
        self.nb_group_labels = len(
            np.unique(self.group_idx)
        )  # number of unique sensitive attribute combinations

    def preprocess(self):
        # Implement any preprocessing steps here if needed
        # data = pd.read_csv(os.path.join(self.data_dir, "data.csv"), index_col=False)
        train_path = os.path.join(self.data_dir, "train.csv")
        test_path = os.path.join(self.data_dir, "test.csv")
        test = pd.read_csv(test_path, index_col=None)
        train = pd.read_csv(train_path, index_col=None)

        # Combine test and train to have data
        data = pd.concat([test, train], ignore_index=True)

        data = data.dropna(
            axis=1, thresh=len(data) * 0.97
        )  # dropping columnns with more that 97% missing values
        data = data.drop(columns=["0", "Ethnicity"])
        # data.dropna(inplace=True)  # dropping rows

        imputer = SimpleImputer(strategy="mean")
        data_values = imputer.fit_transform(data)
        data = pd.DataFrame(data_values, columns=data.columns)
        data = data.apply(pd.to_numeric, errors="coerce")

        # Drop rows with any missing values
        data.dropna(inplace=True)

        # Encode 'num' for binary classification
        self.target = data["target"].astype(int)
        data = data.drop(columns=["target"])

        # Encode 'age' and 'sex'
        data["age"] = data["Age"].apply(lambda x: 0 if x < 60 else 1)
        data["sex"] = data["Gender"].apply(lambda x: 0 if x == 1 else 1)

        data = data.drop(columns=["Age", "Gender"])

        self.process_data_common(data)

        """ Helper function to get group label based on sensitive attributes. Example: if sensitive_attr = ["age", "sex"], then group labels will be 00, 01, 10, 11 """

        return self

    def download(
        self,
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
    ):
        # Downloading is not required as data is expected to be in data_dir
        self.preprocess()


class Mimic(BaseDataModule):
    def __init__(
        self,
        sensitive_attr=["age", "sex"],
        is_1hot=False,
        is_intersectional=False,
        is_decoupled=False,
        *xargs,
        **kwargs,
    ):
        super().__init__(*xargs, **kwargs, model_class=MimicDataset)
        self.model_class_params = {
            "sensitive_attr": sensitive_attr,
            "is_1hot": is_1hot,
            "is_intersectional": is_intersectional,
            "is_decoupled": is_decoupled,
        }

    def get_model_class_params(self):
        return self.model_class_params

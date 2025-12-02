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


class HeartDataset(BaseDataset):
    def __init__(
        self,
        sensitive_attr=["age", "sex"],
        train=False,
        data_dir="./data/heart",
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

    def preprocess(self):
        # Implement any preprocessing steps here if needed
        data = pd.read_csv(os.path.join(self.data_dir, "data.csv"), index_col=False)

        # Replace '?' with NaN and convert to numeric and impute missing values
        data.replace("?", np.nan, inplace=True)
        imputer = SimpleImputer(strategy="mean")
        data_values = imputer.fit_transform(data)
        data = pd.DataFrame(data_values, columns=data.columns)
        data = data.apply(pd.to_numeric, errors="coerce")

        # Drop rows with any missing values
        data.dropna(inplace=True)

        def one_hot_encode(df, column):
            one_hot = pd.get_dummies(data[column], prefix=column, drop_first=True)
            df = df.drop(column, axis=1)
            df = pd.concat([df, one_hot], axis=1)
            return df

        # Ordinal interger encoding for cp, thal, ca, slope and restecg
        """"
        data["ca"] = data["ca"].astype("category").cat.codes
        data["restecg"] = data["restecg"].astype("category").cat.codes
        data["cp"] = data["cp"].astype("category").cat.codes
        data["thal"] = data["thal"].astype("category").cat.codes
        data["slope"] = data["slope"].astype("category").cat.codes
        """

        """ data = one_hot_encode(data, "cp")
        data = one_hot_encode(data, "thal")
        data = one_hot_encode(data, "slope")
        data = one_hot_encode(data, "restecg")
        data = one_hot_encode(data, "ca") """

        # Encode 'num' for binary classification
        self.target = data["num"].apply(lambda x: 1 if x > 0 else 0)

        # Encode 'age' and 'sex'
        data["age"] = data["age"].apply(lambda x: 0 if x < 60 else 1)
        data["sex"] = data["sex"].apply(lambda x: 0 if x == 0.0 else 1)

        """ Helper function to get group label based on sensitive attributes. Example: if sensitive_attr = ["age", "sex"], then group labels will be 00, 01, 10, 11 """

        def _get_group_label(row, columns):
            toRet = ""
            for c in columns:
                # toRet += str(row[c])
                toRet += f"{c}[{int(row[c])}]"
            return toRet

        data = data.drop(columns=["num", "Unnamed: 0"])
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
        print(
            f"Loaded Heart\n List of features: {column_list} \n {'Train' if self.train else 'Test'} \n dataset size: {len(self.train_idx) if self.train else len(self.test_idx)} samples"
        )
        self.input_dim = self.data.shape[1]
        self.group_label = self.group_label.values
        self.target = self.target.values
        self.nb_groups = len(self.sensitive_attr)  # number of sensitive attributes
        self.nb_group_labels = len(
            np.unique(self.group_idx)
        )  # number of unique sensitive attribute combinations

        return self

    def download(
        self,
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
    ):
        file_path = os.path.join(self.data_dir, "data.csv")

        if os.path.exists(file_path):
            self.preprocess()
            return
        columns = [
            "age",
            "sex",
            "cp",
            "trestbps",
            "chol",
            "fbs",
            "restecg",
            "thalach",
            "exang",
            "oldpeak",
            "slope",
            "ca",
            "thal",
            "num",
        ]
        data = pd.read_csv(url, names=columns)
        os.makedirs(self.data_dir, exist_ok=True)
        data.to_csv(file_path)

        self.preprocess()


class Heart(BaseDataModule):
    def __init__(
        self,
        sensitive_attr=["age", "sex"],
        is_1hot=False,
        is_intersectional=False,
        is_decoupled=False,
        *xargs,
        **kwargs,
    ):
        super().__init__(*xargs, **kwargs, model_class=HeartDataset)
        self.model_class_params = {
            "sensitive_attr": sensitive_attr,
            "is_1hot": is_1hot,
            "is_intersectional": is_intersectional,
            "is_decoupled": is_decoupled,
        }

    def get_model_class_params(self):
        return self.model_class_params

from torch.utils.data import Dataset
from fairlearn.datasets import fetch_adult
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from dataloaders.datamodule import BaseDataModule
from dataloaders.dataset import BaseDataset


class AdultDataset(BaseDataset):
    def __init__(
        self,
        sensitive_attr_name="sex",
        train=False,
        seed_everything=42,
    ):
        super().__init__(sensitive_attr_name, train)
        self.sensitive_attr_name = sensitive_attr_name
        self.seed = seed_everything
        self.name = f"adult_{sensitive_attr_name}"

    def __len__(self):
        return len(self.data)

    def preprocess(self):
        # Implement any preprocessing steps here if needed
        adult_data = fetch_adult(as_frame=True, data_home="./data")
        self.data = adult_data.data
        self.target = adult_data.target
        self.sensitive_attr = adult_data.data[
            self.sensitive_attr_name
        ]  # Example sensitive attribute
        self.data = self.data.drop(
            columns=[self.sensitive_attr_name]
        )  # Remove sensitive attribute from features

        if self.sensitive_attr_name == "race":
            idx = adult_data.data["race"].isin(["White", "Black"])
            self.data = self.data[idx]
            self.target = self.target[idx]
            self.sensitive_attr = self.sensitive_attr[idx]
        elif self.sensitive_attr_name == "age":
            self.sensitive_attr = self.sensitive_attr.apply(
                lambda x: 1 if x >= 40 else 0
            )
        # One-hot encode categorical variables

        self.data = pd.get_dummies(self.data, drop_first=True)

        scaler = StandardScaler()
        """ self.data = pd.DataFrame(
            scaler.fit_transform(self.data), columns=self.data.columns
        ) """
        self.data = scaler.fit_transform(self.data).astype("float32")

        y_sc = LabelEncoder()
        self.target = y_sc.fit_transform(self.target)

        s_sc = LabelEncoder()
        self.sensitive_attr = s_sc.fit_transform(self.sensitive_attr)

        self.train_idx, self.test_idx = train_test_split(
            range(len(self.data)),
            test_size=0.2,
            random_state=self.seed,
            stratify=self.target,
        )
        print(
            f"Loaded Adult {'Train' if self.train else 'Test'} dataset: {len(self.train_idx) if self.train else len(self.test_idx)} samples"
        )
        self.input_dim = self.data.shape[1]
        return self

    def download(self):
        adult_data = fetch_adult(as_frame=True, data_home="./data")
        print("Adult dataset downloaded and loaded.")
        print(f"Data shape: {adult_data.data.shape}")
        self.preprocess()


class Adult(BaseDataModule):
    def __init__(self, sensitive_attr_name="sex", *xargs, **kwargs):
        super().__init__(*xargs, **kwargs, model_class=AdultDataset)
        self.model_class_params = {"sensitive_attr_name": sensitive_attr_name}

    def get_model_class_params(self):
        return self.model_class_params

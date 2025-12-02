from torch.utils.data import Dataset
from fairlearn.datasets import fetch_adult
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from dataloaders.datamodule import BaseDataModule
from dataloaders.dataset import BaseDataset
from folktables import (
    ACSDataSource,
    ACSEmployment,
    ACSIncome,
    ACSMobility,
    ACSPublicCoverage,
    ACSTravelTime,
)


class ACSPumsDataset(BaseDataset):
    def __init__(
        self,
        acs_task="income",
        states=["AL"],
        survey_year=2018,
        sensitive_attr_name="sex",
        train=False,
        horizon="1-Year",
        seed_everything=42,
    ):
        super().__init__(sensitive_attr_name, train)
        self.sensitive_attr_name = sensitive_attr_name
        self.acs_task = acs_task
        self.state = states
        self.survey_year = survey_year
        self.horizon = horizon
        self.name = f"{acs_task}_{states}_{sensitive_attr_name}"
        self.seed = seed_everything
        if sensitive_attr_name not in ["SEX", "RAC1P"]:
            raise ValueError(f"Unsupported sensitive attribute: {sensitive_attr_name}")

        if acs_task not in [
            "income",
            "employment",
            "mobility",
            "public_coverage",
            "travel_time",
        ]:
            raise ValueError(f"Unsupported ACS task: {acs_task}")
        # Initialize any required attributes here

    def __len__(self):
        return len(self.data)

    def preprocess(self):

        # Implement any preprocessing steps here if needed
        data_source = ACSDataSource(
            survey_year=self.survey_year, horizon=self.horizon, survey="person"
        )
        acs_data = data_source.get_data(states=self.state, download=True)
        task_class = self._getTask()

        acs_data, self.target, _ = task_class.df_to_pandas(acs_data)
        self.sensitive_attr = acs_data[self.sensitive_attr_name]
        self.sensitive_attr[self.sensitive_attr == 2] - 1  # Map female to 0
        self.target = self.target * 1  # Convert boolean to int (0 and 1)

        self.data = acs_data.drop(
            columns=[self.sensitive_attr_name]
        )  # Remove sensitive attribute from features

        if self.sensitive_attr_name == "RAC1P":
            idx = acs_data.data["RAC1P"].isin([0, 1])
            self.data = self.data[idx]
            self.target = self.target[idx]
            self.sensitive_attr = self.sensitive_attr[idx]
        elif self.sensitive_attr_name == "AGEP":
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
        # adult_data = fetch_adult(as_frame=True, data_home="./data")
        data_source = ACSDataSource(
            survey_year=self.survey_year, horizon=self.horizon, survey="person"
        )
        data_source.get_data(states=self.state, download=True)
        print("ACS PUMS dataset downloaded and loaded.")
        # print(f"Data shape: {adult_data.data.shape}")
        self.preprocess()

    def _getTask(self):
        if self.acs_task == "income":
            return ACSIncome
        elif self.acs_task == "employment":
            return ACSEmployment
        elif self.acs_task == "mobility":
            return ACSMobility
        elif self.acs_task == "public_coverage":
            return ACSPublicCoverage
        elif self.acs_task == "travel_time":
            return ACSTravelTime
        else:
            raise ValueError(f"Unknown ACS task: {self.acs_task}")


class ACSPums(BaseDataModule):
    def __init__(
        self,
        acs_task="income",
        states=["AL"],
        survey_year=2018,
        sensitive_attr_name="sex",
        horizon="1-Year",
        *xargs,
        **kwargs,
    ):
        super().__init__(*xargs, **kwargs, model_class=ACSPumsDataset)
        self.model_class_param = {
            "acs_task": acs_task,
            "states": states,
            "survey_year": survey_year,
            "sensitive_attr_name": sensitive_attr_name,
            "horizon": horizon,
        }

    def get_model_class_params(self):
        return self.model_class_param


if __name__ == "__main__":

    ACSPums()

from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(
        self, sensitive_attr_name, train=False, is_1hot=False, is_intersectional=False
    ):
        self.data = None
        self.target = None
        self.sensitive_attr = None
        self.group_label = None
        self.group_idx = None
        self.train_idx = None
        self.test_idx = None
        self.val_idx = None
        self.train = train
        self.sensitive_attr_name = sensitive_attr_name
        self.input_dim = None  # To be set after preprocessing
        self.is_1hot = is_1hot
        self.is_intersectional = is_intersectional

    def __len__(self):
        return len(self.data)

    def preprocess(self):
        return self

    def download(self):
        pass

    def __getitem__(self, idx):
        sample = self.data[idx, :]
        target = self.target[idx]
        sensitive_attr = self.group_label[idx]
        group_idx = self.group_idx[idx]
        return sample, target, sensitive_attr, group_idx


class SubSet(BaseDataset):
    def __init__(self, dataset: BaseDataset, indices):
        super().__init__(indices)
        self.data = dataset.data[indices, :]
        self.target = dataset.target[indices]
        self.group_label = dataset.group_label[indices]
        self.group_idx = dataset.group_idx[indices]
        self.sensitive_attr = dataset.sensitive_attr
        self.sensitive_attr_name = dataset.sensitive_attr_name

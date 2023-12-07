# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import os
import json
import numpy as np
from tqdm import tqdm
import torch
import torch.utils.data

__all__ = ["AccuracyDataset"]


class RegDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets):
        super(RegDataset, self).__init__()
        self.inputs = inputs
        self.targets = targets

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]

    def __len__(self):
        return self.inputs.size(0)


class AccuracyDataset:
    def __init__(self, path):
        self.path = path
        assert os.path.exists(self.path)
        if not os.path.exists(self.acc_dict_path):
            self.merge_acc_dataset()

    @property
    def acc_src_folder(self):
        return os.path.join(self.path, "src")

    @property
    def acc_dict_path(self):
        return os.path.join(self.path, "acc.dict")

    def merge_acc_dataset(self):
        # load existing data
        merged_acc_dict = []
        for fname in os.listdir(self.acc_src_folder):
            if ".json" not in fname:
                continue
            full_path = os.path.join(self.acc_src_folder, fname)
            partial_acc_dict = json.load(open(full_path))
            merged_acc_dict.extend(partial_acc_dict)
            print("loaded %s" % full_path)
        json.dump(merged_acc_dict, open(self.acc_dict_path, "w"), indent=4)
        return merged_acc_dict

    def build_acc_data_loader(
        self, arch_encoder, n_training_sample=None, batch_size=256, n_workers=16
    ):
        # load data
        acc_dict = json.load(open(self.acc_dict_path))
        X_all = []
        Y_all = []
        with tqdm(total=len(acc_dict), desc="Loading data") as t:
            for (dic, v) in acc_dict:
                X_all.append(arch_encoder.arch2feature(dic))
                Y_all.append(v / 100.0)  # range: 0 - 1
                t.update()
        base_acc = np.mean(Y_all)
        # convert to torch tensor
        X_all = torch.tensor(X_all, dtype=torch.float)
        Y_all = torch.tensor(Y_all)

        # random shuffle
        shuffle_idx = torch.randperm(len(X_all))
        X_all = X_all[shuffle_idx]
        Y_all = Y_all[shuffle_idx]

        # split data
        idx = X_all.size(0) // 5 * 4 if n_training_sample is None else n_training_sample
        val_idx = X_all.size(0) // 5 * 4
        X_train, Y_train = X_all[:idx], Y_all[:idx]
        X_test, Y_test = X_all[val_idx:], Y_all[val_idx:]
        print("Train Size: %d," % len(X_train), "Valid Size: %d" % len(X_test))

        # build data loader
        train_dataset = RegDataset(X_train, Y_train)
        val_dataset = RegDataset(X_test, Y_test)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=False,
            num_workers=n_workers,
        )
        valid_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=False,
            num_workers=n_workers,
        )

        return train_loader, valid_loader, base_acc

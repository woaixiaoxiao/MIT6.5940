# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.


import random
import numpy as np

__all__ = ["MCUNetArchEncoder"]


class MCUNetArchEncoder:
    def __init__(
        self,
        base_depth,
        image_size_list=None,
        width_mult_list=None,
        ks_list=None,
        expand_list=None,
        depth_list=None,
    ):
        self.base_depth = base_depth + [1]
        self.image_size_list = [224] if image_size_list is None else image_size_list
        self.width_mult_list = [1.0] if width_mult_list is None else width_mult_list
        self.ks_list = [3, 5, 7] if ks_list is None else ks_list
        self.expand_list = (
            [3, 4, 6]
            if expand_list is None
            else [int(expand) for expand in expand_list]
        )
        self.depth_list = [0, 1, 2] if depth_list is None else depth_list
        self.n_stage = 6
        # divide between different stages
        self.block_id_to_stage_id = [0 for _ in range(self.max_n_blocks)]
        self.stage_id_to_block_start = [0 for _ in range(len(self.base_depth))]
        num_blocks = 0
        for stage_id, b_d in enumerate(self.base_depth):
            self.stage_id_to_block_start[stage_id] = num_blocks
            # last stage must have depth 1
            if stage_id == len(self.base_depth) - 1:
                cur_depth = b_d
            else:
                cur_depth = b_d + max(self.depth_list)
            for block_id in range(num_blocks, num_blocks + cur_depth):
                self.block_id_to_stage_id[block_id] = stage_id
            num_blocks += cur_depth
        assert num_blocks == self.max_n_blocks
        # build info dict
        self.n_dim = 0
        self.r_info = dict(id2val={}, val2id={}, L=[], R=[])
        self._build_info_dict(target="r")
        self.w_info = dict(id2val={}, val2id={}, L=[], R=[])
        self._build_info_dict(target="w")

        self.k_info = dict(id2val=[], val2id=[], L=[], R=[])
        self.e_info = dict(id2val=[], val2id=[], L=[], R=[])
        self._build_info_dict(target="k")
        self._build_info_dict(target="e")

    @property
    def max_n_blocks(self):
        # last stage must have depth 1
        return (self.n_stage - 1) * max(self.depth_list) + sum(self.base_depth)

    def _build_info_dict(self, target):
        if target == "r":
            target_dict = self.r_info
            target_dict["L"].append(self.n_dim)
            for img_size in self.image_size_list:
                target_dict["val2id"][img_size] = self.n_dim
                target_dict["id2val"][self.n_dim] = img_size
                self.n_dim += 1
            target_dict["R"].append(self.n_dim)
        elif target == "w":
            target_dict = self.w_info
            target_dict["L"].append(self.n_dim)
            for width_mult in range(len(self.width_mult_list)):
                target_dict["val2id"][width_mult] = self.n_dim
                target_dict["id2val"][self.n_dim] = width_mult
                self.n_dim += 1
            target_dict["R"].append(self.n_dim)
        else:
            if target == "k":
                target_dict = self.k_info
                choices = self.ks_list
            elif target == "e":
                target_dict = self.e_info
                choices = self.expand_list
            else:
                raise NotImplementedError
            for i in range(self.max_n_blocks):
                target_dict["val2id"].append({})
                target_dict["id2val"].append({})
                target_dict["L"].append(self.n_dim)
                for k in choices:
                    target_dict["val2id"][i][k] = self.n_dim
                    target_dict["id2val"][i][self.n_dim] = k
                    self.n_dim += 1
                target_dict["R"].append(self.n_dim)

    def arch2feature(self, arch_dict):
        ks, e, d, r, w = (
            arch_dict["ks"],
            arch_dict["e"],
            arch_dict["d"],
            arch_dict["image_size"],
            arch_dict["wid"],
        )

        feature = np.zeros(self.n_dim)
        for i in range(self.max_n_blocks):
            stg = self.block_id_to_stage_id[i]
            nowd = i - self.stage_id_to_block_start[stg]
            # if the block is activated
            if nowd < d[stg] + self.base_depth[stg]:
                # kernel size and expand
                feature[self.k_info["val2id"][i][ks[i]]] = 1
                feature[self.e_info["val2id"][i][e[i]]] = 1
        # resolution
        feature[self.r_info["val2id"][r]] = 1
        # width multiplier
        feature[self.w_info["val2id"][w]] = 1
        return feature

    def feature2arch(self, feature, verbose=False):
        feature_breakdown = []

        img_sz = self.r_info["id2val"][
            int(np.argmax(feature[self.r_info["L"][0] : self.r_info["R"][0]]))
            + self.r_info["L"][0]
        ]
        assert img_sz in self.image_size_list
        if verbose:
            print(
                "image resolution embedding:",
                feature[self.r_info["L"][0] : self.r_info["R"][0]],
                "=> image resolution:",
                img_sz,
            )
        feature_breakdown += feature[self.r_info["L"][0] : self.r_info["R"][0]].tolist()
        feature_breakdown.append("|")

        width_mult = self.w_info["id2val"][
            int(np.argmax(feature[self.w_info["L"][0] : self.w_info["R"][0]]))
            + self.w_info["L"][0]
        ]
        assert width_mult in range(len(self.width_mult_list))
        if verbose:
            print(
                "width multiplier embedding:",
                feature[self.w_info["L"][0] : self.w_info["R"][0]],
                "=> width multiplier:",
                self.width_mult_list[width_mult],
            )
        feature_breakdown += feature[self.w_info["L"][0] : self.w_info["R"][0]].tolist()
        feature_breakdown.append("|")

        arch_dict = {
            "ks": [],
            "e": [],
            "d": [],
            "image_size": img_sz,
            "wid": width_mult,
        }

        d = 0
        for i in range(self.max_n_blocks):
            stg = self.block_id_to_stage_id[i]
            if verbose and i == self.stage_id_to_block_start[stg]:
                print("*" * 50 + f"Stage{stg + 1}" + "*" * 50)

            skip = True
            for j in range(self.k_info["L"][i], self.k_info["R"][i]):
                if feature[j] == 1:
                    arch_dict["ks"].append(self.k_info["id2val"][i][j])
                    skip = False
                    if verbose:
                        print(
                            "kernel size embedding:",
                            feature[self.k_info["L"][i] : self.k_info["R"][i]],
                            "=> kernel size:",
                            self.k_info["id2val"][i][j],
                            end="; ",
                        )
                    feature_breakdown += feature[
                        self.k_info["L"][i] : self.k_info["R"][i]
                    ].tolist()
                    feature_breakdown.append("|")

                    break

            for j in range(self.e_info["L"][i], self.e_info["R"][i]):
                if feature[j] == 1:
                    arch_dict["e"].append(self.e_info["id2val"][i][j])
                    assert not skip
                    skip = False
                    if verbose:
                        print(
                            "expand ratio embedding:",
                            feature[self.e_info["L"][i] : self.e_info["R"][i]],
                            "=> expand ratio:",
                            self.e_info["id2val"][i][j],
                        )
                    feature_breakdown += feature[
                        self.e_info["L"][i] : self.e_info["R"][i]
                    ].tolist()
                    feature_breakdown.append("|")
                    break

            if skip:
                if verbose:
                    print(
                        "kernel size embedding:",
                        feature[self.k_info["L"][i] : self.k_info["R"][i]],
                        "expand ratio embedding:",
                        feature[self.e_info["L"][i] : self.e_info["R"][i]],
                        "=> layer skipped.",
                    )
                feature_breakdown += feature[
                    self.k_info["L"][i] : self.k_info["R"][i]
                ].tolist()
                feature_breakdown.append("|")
                feature_breakdown += feature[
                    self.e_info["L"][i] : self.e_info["R"][i]
                ].tolist()
                feature_breakdown.append("|")
                arch_dict["e"].append(0)
                arch_dict["ks"].append(0)
            else:
                d += 1

            if (i + 1) == self.max_n_blocks:
                arch_dict["d"].append(self.base_depth[stg])
                d = 0
            elif self.block_id_to_stage_id[i + 1] != stg:
                arch_dict["d"].append(d - self.base_depth[stg])
                d = 0
        if not verbose:
            print(
                "network embedding:",
                "[" + " ".join([str(_) for _ in feature_breakdown[:-1]]) + "]",
            )
        return arch_dict

    def random_sample_arch(self):
        return {
            "ks": random.choices(self.ks_list, k=self.max_n_blocks),
            "e": random.choices(self.expand_list, k=self.max_n_blocks),
            "d": random.choices(self.depth_list, k=self.n_stage),
            "image_size": random.choice(self.image_size_list),
            "wid": random.choice(range(len(self.width_mult_list))),
        }

    def mutate_resolution(self, arch_dict, mutate_prob):
        if random.random() < mutate_prob:
            arch_dict["image_size"] = random.choice(self.image_size_list)
        return arch_dict

    def mutate_width(self, arch_dict, mutate_prob):
        if random.random() < mutate_prob:
            arch_dict["wid"] = random.choice(range(len(self.width_mult_list)))
        return arch_dict

    def mutate_arch(self, arch_dict, mutate_prob):
        for i in range(self.max_n_blocks):
            if random.random() < mutate_prob:
                arch_dict["ks"][i] = random.choice(self.ks_list)
                arch_dict["e"][i] = random.choice(self.expand_list)

        for i in range(self.n_stage):
            if random.random() < mutate_prob:
                arch_dict["d"][i] = random.choice(self.depth_list)
        return arch_dict

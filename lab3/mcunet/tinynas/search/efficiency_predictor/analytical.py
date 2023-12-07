import json
import os
import torch
from mcunet.utils.pytorch_utils import count_peak_activation_size, count_net_flops


class AnalyticalEfficiencyPredictor:
    def __init__(self, net):
        self.net = net

    def get_efficiency(self, spec: dict):
        self.net.set_active_subnet(**spec)
        data_shape = (1, 3, spec["image_size"], spec["image_size"])
        subnet = self.net.get_active_subnet()
        if torch.cuda.is_available():
            subnet = subnet.cuda()
        flops = count_net_flops(subnet, data_shape)
        peak_memory = count_peak_activation_size(subnet, data_shape)
        return dict(flops=flops / 1e6, peak_memory=peak_memory / 1024)

    def satisfy_constraint(self, measured: dict, target: dict):
        for key in measured:
            assert key in target
            if measured[key] > target[key]:
                return False
        return True

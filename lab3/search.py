import argparse
import json
from tqdm import tqdm
from mcunet.tinynas.search.efficiency_predictor.analytical import (
    AnalyticalEfficiencyPredictor,
)
import numpy as np
import os
import torch
from mcunet.tinynas.search.accuracy_predictor import (
    AccuracyDataset,
    AccuracyPredictor,
    MCUNetArchEncoder,
)

# from mcunet.tinynas.search.efficiency_predictor import MACSPredictor
from mcunet.tinynas.search.arch_searcher import EvolutionSearcher
from mcunet.tinynas.elastic_nn.networks.ofa_proxyless_w import OFAProxylessNASNets

from mcunet.utils.mcunet_eval_helper import build_val_data_loader, calib_bn, validate


def main():
    data_dir = "data/vww-s256/val"
    device = "cuda:0"

    ofa_network = OFAProxylessNASNets(
        n_classes=2,
        bn_param=(0.1, 1e-3),
        dropout_rate=0.0,
        base_stage_width="proxyless384",
        width_mult_list=[0.5, 0.75, 1.0],
        ks_list=[3, 5, 7],
        expand_ratio_list=[3, 4, 6],
        depth_list=[0, 1, 2],
        base_depth=[1, 2, 2, 2, 2],
        fuse_blk1=True,
        se_stages=[False, [False, True, True, True], True, True, True, False],
    )

    ofa_network.load_state_dict(
        torch.load("vww_supernet.pth", map_location="cpu")["state_dict"], strict=True
    )

    ofa_network = ofa_network.cuda()

    image_size_list = [96, 112, 128, 144, 160]
    arch_encoder = MCUNetArchEncoder(
        image_size_list=image_size_list,
        base_depth=ofa_network.base_depth,
        depth_list=ofa_network.depth_list,
        expand_list=ofa_network.expand_ratio_list,
        width_mult_list=ofa_network.width_mult_list,
    )
    os.makedirs("pretrained", exist_ok=True)
    acc_pred_checkpoint_path = (
        f"pretrained/{ofa_network.__class__.__name__}_acc_predictor.pth"
    )
    acc_predictor = AccuracyPredictor(
        arch_encoder,
        400,
        3,
        checkpoint_path=acc_pred_checkpoint_path
        if os.path.exists(acc_pred_checkpoint_path)
        else None,
        device=device,
    )
    efficiency_predictor = AnalyticalEfficiencyPredictor(ofa_network)
    nas_agent = EvolutionSearcher(
        efficiency_predictor, acc_predictor, population_size=30, max_time_budget=10
    )

    # train accuracy predictor
    if not os.path.exists(acc_pred_checkpoint_path):
        acc_dataset = AccuracyDataset("acc_datasets")
        train_loader, valid_loader, base_acc = acc_dataset.build_acc_data_loader(
            arch_encoder=arch_encoder
        )
        criterion = torch.nn.L1Loss().to(device)
        optimizer = torch.optim.Adam(acc_predictor.parameters())
        # the default value is zero
        acc_predictor.base_acc.data += base_acc
        for epoch in tqdm(range(10)):
            acc_predictor.train()
            for (data, label) in tqdm(train_loader, desc="Train Epoch%d" % (epoch + 1)):
                data = data.to(device)
                label = label.to(device)
                pred = acc_predictor(data)
                loss = criterion(pred, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            acc_predictor.eval()
            with torch.no_grad():
                with tqdm(total=len(valid_loader), desc="Validate") as t:
                    for (data, label) in valid_loader:
                        data = data.to(device)
                        label = label.to(device)
                        pred = acc_predictor(data)
                        loss = criterion(pred, label)
                        t.set_postfix({"loss": loss.item()})
                        t.update(1)
        torch.save(acc_predictor.cpu().state_dict(), acc_pred_checkpoint_path)

    # search
    search_constraint = dict(flops=100.0, peak_memory=256.0)
    best_valids, best_info = nas_agent.run_evolution_search(
        constraint=search_constraint, verbose=True
    )
    print(best_valids, best_info)

    # validate
    subnet = ofa_network.get_active_subnet().cuda()
    calib_bn(subnet, data_dir, 128, best_info[1]["image_size"])
    val_loader = build_val_data_loader(data_dir, best_info[1]["image_size"], 128)
    acc = validate(subnet, val_loader)
    print(acc)


if __name__ == "__main__":
    main()

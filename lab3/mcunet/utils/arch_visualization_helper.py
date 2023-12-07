import json, os, os.path as osp
import pickle

import numpy as np
import matplotlib.pyplot as plt

from graphviz import Digraph

k3c = "#%02x%02x%02x" % (0, 118, 197)  # "blue"
k5c = "#%02x%02x%02x" % (247, 173, 40)  # yellow
k7c = "#%02x%02x%02x" % (153, 20, 0)  # "red"

c_lut = {3: k3c, 5: k5c, 7: k7c}

w_lut = {3: 3, 4: 3.5, 6: 4.5}


def draw_arch(ks_list, ex_list, d_list, resolution, out_name="viz/temp", info=None):

    stage_id_to_block_start = [0, 3, 7, 11, 15, 19]
    block_id_to_stage_id = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5]
    base_depth = [1, 2, 2, 2, 2, 1]
    strides = [2, 2, 2, 1, 2, 1]
    resolutions = []
    cur_resolution = resolution // 2
    for idx in range(len(strides)):
        cur_resolution //= strides[idx]
        resolutions.append(cur_resolution)

    ddot = Digraph(
        comment="The visualization of Mojito Architecture Search",
        format="png",
        graph_attr={"size": "20,60"},
        node_attr={"fontsize": "32", "height": "0.8"},
    )
    model_name = "mojito"
    node_cnt = 0
    with ddot.subgraph(name=model_name) as dot:
        dot.node(
            "%s-%s" % (model_name, node_cnt),
            "stage 1",
            fontcolor="black",
            style="rounded,filled",
            shape="record",
            color="lightgray",
            width=str(4.5),
        )
        prev = 0
        node_cnt += 1

        for idx in range(20):

            ks = ks_list[idx]
            ex = ex_list[idx]
            stage_id = block_id_to_stage_id[idx]
            stage_depth = d_list[stage_id]

            if idx == stage_id_to_block_start[stage_id] and stage_id > 0:
                dot.node(
                    "%s-%s" % (model_name, node_cnt),
                    "stage %d" % (stage_id + 1),
                    fontcolor="black",
                    style="rounded,filled",
                    shape="record",
                    color="lightgray",
                    width=str(4.5),
                )
                dot.edge(
                    "%s-%s" % (model_name, prev),
                    "%s-%s" % (model_name, node_cnt),
                    fontcolor="black",
                    label=f'<<FONT POINT-SIZE="32">{resolutions[stage_id-1]}x{resolutions[stage_id-1]}</FONT>>',
                )
                prev = node_cnt
                node_cnt += 1

            if (
                idx - stage_id_to_block_start[stage_id]
                >= stage_depth + base_depth[stage_id]
            ):
                continue

            # if ks == 0 and ex == 0:
            #    # print("Skipped")
            #    continue
            else:
                pass
                # print(w_lut[ex])

            new_name = f"MBConv{ex}-{ks}x{ks}"
            dot.node(
                "%s-%s" % (model_name, node_cnt),
                new_name,
                fontcolor="white",
                style="rounded,filled",
                shape="record",
                color=c_lut[ks],
                width=str(w_lut[ex]),
            )
            if prev is not None:
                if prev == 0:
                    cur_res = resolution // 2
                elif idx == stage_id_to_block_start[stage_id] and stage_id > 0:
                    cur_res = resolutions[stage_id - 1]
                else:
                    cur_res = resolutions[stage_id]
                dot.edge(
                    "%s-%s" % (model_name, prev),
                    "%s-%s" % (model_name, node_cnt),
                    fontcolor="black",
                    label=f'<<FONT POINT-SIZE="32">{cur_res}x{cur_res}</FONT>>',
                )
            prev = node_cnt
            node_cnt += 1

    if info is not None:
        res = []
        for k, v in info.items():
            res.append("%s: %.2f" % (k, v))
        result = " ".join(res)
        ddot.attr(label=f'<<FONT POINT-SIZE="32">{result}</FONT>>', labelloc="top")

    os.makedirs(osp.dirname(out_name), exist_ok=True)
    ddot.render(out_name)
    # print(f"The arch is visualized to {out_name}")

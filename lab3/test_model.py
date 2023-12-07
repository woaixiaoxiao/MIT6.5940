import torch
from mcunet.tinynas.elastic_nn.networks.ofa_proxyless_w import (
    OFAProxylessNASNets as OFAProxylessNASNetsW,
)


def parse_supernet_config(ks_list, expand_list, depth_list, width_list, verbose=True):
    if isinstance(ks_list, str):
        ks_list = [int(s) for s in ks_list.split(",")]
    if isinstance(expand_list, str):
        expand_list = [int(s) for s in expand_list.split(",")]
    if isinstance(depth_list, str):
        depth_list = [int(s) for s in depth_list.split(",")]
    if isinstance(width_list, str):
        width_list = [float(s) for s in width_list.split(",")]
    if verbose:
        print(" *** OFA config:")
        print(" * ks:\t\t", ks_list)
        print(" * expand\t\t", expand_list)
        print(" * depth\t\t", depth_list)
        print(" * width\t\t", width_list)
    return ks_list, expand_list, depth_list, width_list


def get_supernet(
    arch,
    n_class,
    width_list,
    ks_list,
    expand_list,
    depth_list,
    dropout=0.0,
    verbose=True,
):
    ks_list, expand_list, depth_list, width_list = parse_supernet_config(
        ks_list, expand_list, depth_list, width_list, verbose=verbose
    )
    assert arch == "ofaproxyless384"
    model = OFAProxylessNASNetsW(
        n_classes=n_class,
        bn_param=(0.1, 1e-3),
        dropout_rate=dropout,
        base_stage_width="proxyless384",
        width_mult_list=width_list,
        ks_list=ks_list,
        expand_ratio_list=expand_list,
        depth_list=depth_list,
        base_depth=(1, 2, 2, 2, 2),
        fuse_blk1=True,
        se_stages=[False, [False, True, True, True], True, True, True, False],
    )
    return model


supernet = get_supernet(
    arch="ofaproxyless384",
    n_class=2,
    width_list="0.5,0.75,1.0",
    ks_list="3,5,7",
    expand_list="3,4,6",
    depth_list="0,1,2",
)

# print(supernet)
cnt = 0
num_param = 0
for name, param in supernet.named_parameters():
    # print(name, param.shape, param.numel())
    cnt += param.numel()
    num_param += 1
print(cnt, num_param)

cp = torch.load(
    "../OFA_ofaproxyless384_vww_cosine_e10_dbatch4_r96,256_w0.5,1.0_k3,5,7_e3,4,6_d2,3,4_valid5000_adaw/ckpt.pth",
    map_location="cpu",
)
model = cp["state_dict"]
new_model = dict()
cnt = 0
num_param = 0
# print("\n" * 10)
for key in model:
    if "blocks" in key:
        new_key = key.split(".")
        new_key[2] = "mobile_inverted_conv"
        new_key = ".".join(new_key)
        new_model[new_key] = model[key]
    else:
        new_model[key] = model[key]
    if "running" in key or "track" in key:
        continue
    # print(key, model[key].shape, model[key].numel())
    cnt += model[key].numel()
    num_param += 1

print(cnt, num_param)
supernet.load_state_dict(new_model)
print(supernet.sample_active_subnet())

cp = dict(state_dict=new_model)
torch.save(cp, "vww_supernet.pth")

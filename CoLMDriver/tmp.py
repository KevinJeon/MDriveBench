#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import os
import random

import numpy as np
import torch

from common.io import load_config_from_yaml
from common.random import set_random_seeds
from common.registry import build_object_within_registry_from_config
from common.torch_helper import load_checkpoint

from codriving import CODRIVING_REGISTRY
from codriving.models.model_decoration import decorate_model
from codriving.utils import initialize_root_logger
from codriving.utils.torch_helper import build_dataloader, move_dict_data_to_device

from common.detection import warp_image

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils


def parse_args():
    parser = argparse.ArgumentParser(description="Eval-only debug runner")
    parser.add_argument(
        "-c",
        "--config-file",
        required=True,
        type=str,
        help="Config file used in train_end2end.py",
    )
    parser.add_argument(
        "--out-dir",
        default="./tmp_eval_output",
        type=str,
        help="Directory to output eval logs",
    )
    parser.add_argument(
        "--log-filename",
        default="tmp_eval",
        type=str,
        help="Log filename without extension",
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        help="Planner checkpoint path (epoch_*.ckpt)",
    )
    parser.add_argument(
        "--model_dir",
        default="",
        type=str,
        help="Perception checkpoint dir/path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=73,
        help="Random seed",
    )
    return parser.parse_args()


def build_device():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        return local_rank, torch.device(f"cuda:{local_rank}")
    return local_rank, torch.device("cpu")


@torch.no_grad()
def validate_only(pred_model, perce_model, loss_func, dataloader, device):
    pred_model.eval()
    perce_model.eval()
    all_losses = []

    for batch_idx, batch_data in enumerate(dataloader):
        pred_batch_data, perce_batch_data_dict = batch_data
        move_dict_data_to_device(pred_batch_data, device)

        pred_batch_data.update(
            {
                "fused_feature": [],
                "features_before_fusion": [],
            }
        )

        frame_list = list(perce_batch_data_dict.keys())
        frame_list.sort()
        perception_results_list = []

        for frame in frame_list:
            perce_batch_data_dict[frame] = train_utils.to_device(
                perce_batch_data_dict[frame], device
            )
            perception_results = perce_model(perce_batch_data_dict[frame]["ego"])

            fused_feature_2 = perception_results["fused_feature"].permute(0, 1, 3, 2)
            fused_feature_3 = torch.flip(fused_feature_2, dims=[2])
            pred_batch_data["fused_feature"].append(fused_feature_3[:, :, :192, :])
            perception_results_list.append(perception_results)

        pred_batch_data["feature_warpped_list"] = []
        for b in range(len(perception_results_list[0]["fused_feature"])):
            feature_dim = perception_results_list[0]["fused_feature"].shape[1]
            feature_to_warp = torch.zeros(1, 5, feature_dim, 192, 96).to(device).float()
            det_map_pose = torch.zeros(1, 5, 3).to(device).float()

            for t in range(5):
                feature_to_warp[0, t, :] = pred_batch_data["fused_feature"][t][b]
                det_map_pose[:, t] = torch.tensor(pred_batch_data["detmap_pose"][b, t]).to(
                    device
                )

            feature_warped = warp_image(det_map_pose, feature_to_warp)
            pred_batch_data["feature_warpped_list"].append(feature_warped)

        model_output = pred_model(pred_batch_data)
        loss, _extra_info = loss_func(pred_batch_data, model_output)
        all_losses.append(loss.detach().cpu())

        if batch_idx % 10 == 0:
            logging.info("eval iter %d/%d loss=%s", batch_idx, len(dataloader), float(loss))

    mean_loss = torch.stack(all_losses).mean()
    logging.info("Eval done. mean_loss=%s", float(mean_loss))
    print(f"Eval done. mean_loss={float(mean_loss):.6f}")


def main():
    args = parse_args()
    if args.resume == "None":
        args.resume = ""

    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1234)

    local_rank, device = build_device()
    set_random_seeds(args.seed, local_rank)

    os.makedirs(args.out_dir, exist_ok=True)
    initialize_root_logger(path=f"{args.out_dir}/{args.log_filename}.txt")
    logging.info("Using device: %s", device)

    perception_hypes = yaml_utils.load_yaml(None, args)
    config = load_config_from_yaml(args.config_file)

    val_data_config = config["data"]["validation"]
    val_data_config["dataset"]["perception_hypes"] = perception_hypes
    val_dataloader = build_dataloader(val_data_config, distributed=False)

    perception_model = train_utils.create_model(perception_hypes)
    if args.model_dir:
        _epoch, perception_model = train_utils.load_saved_model(args.model_dir, perception_model)
    perception_model.to(device)

    model = build_object_within_registry_from_config(
        CODRIVING_REGISTRY,
        config["model"],
    )
    decorate_model(model, **config["model_decoration"])
    model.to(device)

    loss_func = build_object_within_registry_from_config(
        CODRIVING_REGISTRY,
        config["loss"],
    )

    if args.resume:
        load_checkpoint(args.resume, device, model, strict=False)
        logging.info("Loaded planner checkpoint: %s", args.resume)
    else:
        logging.warning("No planner checkpoint given; running eval with random planner weights.")

    validate_only(model, perception_model, loss_func, val_dataloader, device)


if __name__ == "__main__":
    main()

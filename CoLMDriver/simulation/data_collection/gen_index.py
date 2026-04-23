import json
import os
import sys
from tqdm import tqdm
import time

data_index = ""
frame_all = 0
sample_all = 0
route_num = {}
town_num = {}

dataset_directory = "dataset"


def _route_measurements_flat_speed_ok(sub_path: str, ego_list: list) -> bool:
    """
    Training (pnp_dataset_*) expects measurements/*.json with top-level 'speed'.
    Skip routes where speed only lives under move_state (newer V2Xverse export).
    """
    if not ego_list:
        return False
    for ego in sorted(ego_list):
        mdir = os.path.join(sub_path, ego, "measurements")
        if not os.path.isdir(mdir):
            return False
        jsons = sorted(f for f in os.listdir(mdir) if f.endswith(".json"))
        if not jsons:
            return False
        try:
            with open(os.path.join(mdir, jsons[0]), encoding="utf-8") as fp:
                data = json.load(fp)
        except (OSError, json.JSONDecodeError):
            return False
        if not isinstance(data, dict) or "speed" not in data:
            return False
    return True


for i in range(1):
    subs = os.listdir(os.path.join(dataset_directory, "weather-%d/data" % i))
    for sub in tqdm(subs):
        seq_len = 1000000
        sub_path = os.path.join(dataset_directory, "weather-{}/data/{}/".format(i, sub))
        try:
            agent_list = os.listdir(sub_path)
        except Exception:
            continue
        ego_list = [ego for ego in agent_list if ego.startswith("ego")]
        rsu_list = [ego for ego in agent_list if ego.startswith("rsu")]
        if not _route_measurements_flat_speed_ok(sub_path, ego_list):
            continue
        for ego in ego_list:
            ego_path = os.path.join(sub_path, ego)
            seq_len_ego = len(os.listdir(os.path.join(ego_path, "rgb_front")))
            if seq_len > seq_len_ego:
                seq_len = seq_len_ego
        if seq_len > 50:
            if len(ego_list) == 1 and len(rsu_list) == 0:
                continue
            # data_index += "{} {} {}\n".format(sub_path, seq_len, len(ego_list))
            # frame_all += seq_len
            # sample_all += seq_len*len(ego_list)

            town_route_id = sub.split("_")[1] + "_" + sub.split("_")[2]
            if town_route_id not in route_num:
                town = int(sub.split("_")[1][-2:])
                # if town not in [7,10]:
                #     continue
                route_num[town_route_id] = {
                    "seq_len": seq_len,
                    "sub_path": sub_path,
                    "len(ego_list)": len(ego_list),
                }
                if town not in town_num:
                    town_num[town] = 1
                else:
                    town_num[town] += 1
            elif route_num[town_route_id]["seq_len"] < seq_len:
                route_num[town_route_id] = {
                    "seq_len": seq_len,
                    "sub_path": sub_path,
                    "len(ego_list)": len(ego_list),
                }


exist_path = []
print(len(exist_path))
a = 0

with open(os.path.join(dataset_directory, "dataset_index.txt"), "w") as f:
    for town_route_id in route_num:

        if route_num[town_route_id]["sub_path"] in exist_path:
            continue
            # time.sleep(1000)
        route_num[town_route_id]["seq_len"] -= 25
        data_index += "{} {} {}\n".format(
            route_num[town_route_id]["sub_path"],
            route_num[town_route_id]["seq_len"],
            route_num[town_route_id]["len(ego_list)"],
        )
        seq_len = route_num[town_route_id]["seq_len"]
        frame_all += seq_len
        sample_all += seq_len * route_num[town_route_id]["len(ego_list)"]
    f.write(data_index)
print("frames:", frame_all)
print("samples:", sample_all)
print(town_num)

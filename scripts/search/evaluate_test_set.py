import numpy as np
from basic_traffic_env import BasicTrafficEnv
import json
from pathlib import Path
import os
from env_utils import (
    get_videos_of_small_ttc_cases,
    multiprocessing_search_pso,
    analyze_results,
)
import argparse
import time
from datetime import datetime


from front_brake import FrontBrake
from front_cut_in_with_one_npc import FrontCutInWithOneNPC
from front_cut_in_with_two_npc import FrontCutInWithTwoNPC
from opposite_vehicle_taking_priority import OppositeVehicleTakingPriority
from nonsignalized_junction_left_turn import NonSignalizedJunctionLeftTurn
from nonsignalized_junction_right_turn import NonSignalizedJunctionRightTurn
import pickle


# main


def main(name, w, agent, params_list, config="scripts/config.json", random_seed=0):
    # read config, get config where name==name
    with open(config, "r") as f:
        config = json.load(f)
    for scene in config["scenarios"]:
        if scene["name"] == name:
            scene_config = scene
            break

    if scene_config is None:
        raise ValueError("scene config is none")

    # import class by scene_config['class_name']
    class_name = scene_config["class_name"]
    class_ = globals()[class_name]

    name = scene_config["name"]
    env_name = scene_config["env_name"]
    model_path = Path(scene_config["model_path"])
    natural_data_path = Path(scene_config["natural_data_path"])
    params_dict = scene_config["params_dict"]

    w = w
    agent = agent
    random_seed = random_seed
    formatted_time = datetime.now().strftime("%m%d%H%M")
    checkpoint = False

    output_path = Path(f"output-eva-{agent}/{name}/{formatted_time}_{w}_{random_seed}")

    ttc_mode = scene_config["ttc_mode"]

    # Init env
    # debug=True will print step info
    scenarioInstance = class_(
        env_name=env_name,
        render=False,
        model_path=model_path,
        natural_data_path=natural_data_path,
        output_path=output_path,
        name=name,
        debug=False,
        random_seed=random_seed,
        agent=agent,
        checkpoint=checkpoint,
    )

    scenarioInstance.set_save_data(True)  # Save output data
    scenarioInstance.set_save_video(False)  # Save simulator video
    scenarioInstance.set_render(False)  # Render simulator
    scenarioInstance.set_debug(False)  # Print debug

    scenarioInstance.setup(
        params_dict,
        loss_weight=[w, 1 - w, 50],
        ttc_mode=ttc_mode,
    )

    print(f"======Start run {name} with replay, save at {output_path}======")

    scenarioInstance.replay(params_list)

    # remove the instance
    del scenarioInstance


if __name__ == "__main__":

    TEST_SET_PATH = (
        "/home/tay/Workspace/highway-scen-gen/scripts/data_ana/test_set_apollo.pkl"
    )
    AGENT = "apollo"

    test_set = None
    with open(TEST_SET_PATH, "rb") as f:
        test_set = pickle.load(f)

    for scenario, w in test_set.items():
        if 'right' not in scenario:
            continue
        for k, v in w.items():
            if k == 0.3:
                for seed in [2,4,6,8,10,999,666,5020,8099,213,10086,1213,319]:
                    main(scenario, k, AGENT, v, random_seed=seed)

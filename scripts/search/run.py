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


# main


def main(name, w, agent, config="scripts/config.json", random_seed=0, method="spso", output_base="output"):
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
    n_particles = 20
    max_iter = 50
    agent = agent
    random_seed = random_seed
    formatted_time = datetime.now().strftime("%m%d%H%M")
    checkpoint = False

    output_path = Path(
        f"{output_base}/{name}/{agent}/{formatted_time}_{w}_{n_particles}x{max_iter}_{random_seed}"
    )

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

    print(f"======Start search {name} with SPSO, save at {output_path}======")

    if method == "spso":
        scenarioInstance.search_spso(n_particles=n_particles, max_iter=max_iter)
    if method == "ga":
        scenarioInstance.search_ga(n_population=n_particles, n_generation=max_iter, prob_mut=0.1)

    # remove the instance
    del scenarioInstance


if __name__ == "__main__":

    scenarios = [
        "front_brake",
        "front_cut_in_with_one_npc",
        "front_cut_in_with_two_npc",
        "opposite_vehicle_taking_priority",
        "nonsignalized_junction_left_turn",
        "nonsignalized_junction_right_turn",
    ]

    for s in scenarios:
        main(name=s, w=0.05, agent="apollo", method="ga", output_base="output-ga-hw")

    for s in scenarios:
        main(name=s, w=0.21, agent="ba", method="ga", output_base="output-ga-hw")

    for s in scenarios:
        main(name=s, w=0.37, agent="tfpp", method="ga", output_base="output-ga-hw")

    for s in scenarios:
        main(name=s, w=0.29, agent="interfuser", method="ga", output_base="output-ga-hw")

    for s in scenarios:
        main(name=s, w=0.45, agent="highway", method="ga", output_base="output-ga-hw")
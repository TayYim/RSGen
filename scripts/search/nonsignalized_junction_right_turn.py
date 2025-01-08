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
from datetime import datetime


class NonSignalizedJunctionRightTurn(BasicTrafficEnv):

    def _process_natural_data(self):
        """
        Process natural data
        """
        with open(self.natural_data_path, "r") as infile:
            sr = infile.read()
            hb_vehicles = json.loads(sr)
        decision_var = []
        for hb_vehicle in hb_vehicles:
            v_ego = hb_vehicle["start_info"]["v_ego"]
            v_1 = hb_vehicle["start_info"]["v_1"]
            r_ego = hb_vehicle["start_info"]["r_ego"]
            r_1 = hb_vehicle["start_info"]["r_1"]
            decision_var.append([r_ego, v_ego, r_1, v_1])

        # Save result
        self.decision_var = np.array(decision_var)

    # Override in subclass. Order sensitive

    def _calculate_similarity(self):

        inputs = [
            self.search_collector["r_ego"][-1],
            self.search_collector["v_ego"][-1],
            self.search_collector["r_1"][-1],
            self.search_collector["v_1"][-1],
        ]

        similarity = super()._calculate_similarity(inputs)

        return similarity

    def run_step(self, r_ego, v_ego, r_1, v_1):

        self.env.unwrapped.config["r_ego"] = r_ego
        self.env.unwrapped.config["v_ego"] = v_ego
        self.env.unwrapped.config["r_1"] = r_1
        self.env.unwrapped.config["v_1"] = v_1

        self.search_collector["r_ego"].append(r_ego)
        self.search_collector["r_1"].append(r_1)
        self.search_collector["v_ego"].append(v_ego)
        self.search_collector["v_1"].append(v_1)

        return super().run_step()

    @BasicTrafficEnv.search_method_decorator("natural")
    def search_natural(self):

        total_len = len(self.decision_var)

        for i in range(total_len):
            r_ego = self.decision_var[i][0]
            v_ego = self.decision_var[i][1]
            r_1 = self.decision_var[i][2]
            v_1 = self.decision_var[i][3]
            if (
                self.within_bounds("r_1", r_1)
                and self.within_bounds("r_ego", r_ego)
                and self.within_bounds("v_ego", v_ego)
                and self.within_bounds("v_1", v_1)
            ):
                self.run_step(r_ego, v_ego, r_1, v_1)


if __name__ == "__main__":

    # -- Init params --
    name = "nonsignalized_junction_right_turn"
    w = 0.0
    n_particles = 20
    max_iter = 50
    agent = "apollo"
    random_seed = 0
    formatted_time = datetime.now().strftime("%m%d%H%M")
    checkpoint = False

    output_path = Path(
        f"output/{name}/{agent}/{formatted_time}_{w}_{n_particles}x{max_iter}_{random_seed}"
    )

    ## Checkpoint
    # max_iter = 5
    checkpoint = True
    output_path = Path("output/nonsignalized_junction_right_turn/apollo/03300825_0.0_20x50_0")

    # Init env
    # debug=True will print step info
    scenarioInstance = NonSignalizedJunctionRightTurn(
        env_name="NonSignalizedJunctionRightTurn-v0",
        render=False,
        model_path=Path(
            "data/models/MAF_crossroads_ego_right_turn_bv_on_left_best.pth"
        ),
        natural_data_path=Path(
            "data/datasets/crossroads_ego_right_turn_bv_on_left.json"
        ),
        output_path=output_path,
        name=name,
        debug=False,
        random_seed=random_seed,
        agent=agent,
        checkpoint=checkpoint,
    )

    # Order of parameters should be consistent with the order of parameters in run_step()
    params_dict = {
        "r_ego": [10.027, 69.495],
        "v_ego": [0.0, 15.167],
        "r_1": [10.13, 68.432],
        "v_1": [0.0, 20.559],
    }

    scenarioInstance.set_save_data(True)  # Save output data
    scenarioInstance.set_save_video(False)  # Save simulator video
    scenarioInstance.set_render(False)  # Render simulator
    scenarioInstance.set_debug(False)  # Print debug

    scenarioInstance.setup(
        params_dict,
        loss_weight=[w, 1 - w, 50],
        ttc_mode="cross",
    )

    scenarioInstance.search_spso(n_particles=n_particles, max_iter=max_iter)

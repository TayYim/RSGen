import json
from basic_traffic_env import BasicTrafficEnv
import numpy as np
from pathlib import Path
import os
from env_utils import (
    get_videos_of_small_ttc_cases,
    multiprocessing_search_pso,
    analyze_results,
)
from datetime import datetime


class FrontCutInWithTwoNPC(BasicTrafficEnv):

    def _process_natural_data(self):
        """
        Process natural data
        """
        with open(self.natural_data_path, "r") as infile:
            sr = infile.read()
            vehicles = json.loads(sr)
        decision_var = []
        for vehicle in vehicles:
            speed = vehicle["start_info"]["ego"]["speed"] * 0.3048
            range_1 = vehicle["start_info"]["range_1"] * 0.3048
            range_rate_1 = vehicle["start_info"]["range_rate_1"] * 0.3048
            range_2 = vehicle["start_info"]["range_2"] * 0.3048
            range_rate_2 = vehicle["start_info"]["range_rate_2"] * 0.3048
            decision_var.append([speed, range_1, range_rate_1, range_2, range_rate_2])

        # Save result
        self.decision_var = np.array(decision_var)

    # Override in subclass. Order sensitive

    def _calculate_similarity(self):

        inputs = [
            self.search_collector["absolute_v"][-1],
            self.search_collector["relative_p_1"][-1],
            self.search_collector["relative_v_1"][-1],
            self.search_collector["relative_p_2"][-1],
            self.search_collector["relative_v_2"][-1],
        ]

        similarity = super()._calculate_similarity(inputs)

        return similarity

    def run_step(
        self, absolute_v, relative_p_1, relative_v_1, relative_p_2, relative_v_2
    ):

        self.env.unwrapped.config["absolute_v"] = absolute_v
        self.env.unwrapped.config["relative_p_1"] = relative_p_1
        self.env.unwrapped.config["relative_v_1"] = relative_v_1
        self.env.unwrapped.config["relative_p_2"] = relative_p_2
        self.env.unwrapped.config["relative_v_2"] = relative_v_2

        self.search_collector["absolute_v"].append(absolute_v)
        self.search_collector["relative_v_1"].append(relative_v_1)
        self.search_collector["relative_p_1"].append(relative_p_1)
        self.search_collector["relative_v_2"].append(relative_v_2)
        self.search_collector["relative_p_2"].append(relative_p_2)

        return super().run_step()

    @BasicTrafficEnv.search_method_decorator("natural")
    def search_natural(self):

        total_len = len(self.decision_var)

        for i in range(total_len):
            absolute_v = self.decision_var[i][0]
            relative_p_1 = self.decision_var[i][1]
            relative_v_1 = self.decision_var[i][2]
            relative_p_2 = self.decision_var[i][3]
            relative_v_2 = self.decision_var[i][4]
            if (
                self.within_bounds("absolute_v", absolute_v)
                and self.within_bounds("relative_p_1", relative_p_1)
                and self.within_bounds("relative_v_1", relative_v_1)
                and self.within_bounds("relative_p_2", relative_p_2)
                and self.within_bounds("relative_v_2", relative_v_2)
            ):
                self.run_step(
                    absolute_v, relative_p_1, relative_v_1, relative_p_2, relative_v_2
                )


if __name__ == "__main__":

    # -- Init params --
    name = "front_cut_in_with_two_npc"
    w = 0.5
    n_particles = 1
    max_iter = 1
    agent = "apollo"
    random_seed = 0
    formatted_time = datetime.now().strftime("%m%d%H%M")
    checkpoint = False

    output_path = Path(
        f"output/{name}/{agent}/{formatted_time}_{w}_{n_particles}x{max_iter}_{random_seed}"
    )

    ## Checkpoint
    # max_iter = 50
    # checkpoint = True
    # output_path = Path("output-checkpoint/front_cut_in_with_two_npc/tfpp/04012315_0.0_20x50_0")

    # Init env
    # debug=True will print step info
    scenarioInstance = FrontCutInWithTwoNPC(
        env_name="front-cut-in-with-two-npc-v0",
        render=False,
        model_path=Path("data/models/MAF_front_cut_in_with_two_npcs_best.pth"),
        natural_data_path=Path("data/datasets/front_cut_in_with_two_npcs.json"),
        output_path=output_path,
        name=name,
        debug=False,
        random_seed=random_seed,
        agent=agent,
        checkpoint=checkpoint,
    )

    # Order of parameters should be consistent with the order of parameters in run_step()
    params_dict = {
        "absolute_v": [0, 21.214],
        "relative_p_1": [0.238, 69.291],
        "relative_v_1": [-8.163, 9.373],
        "relative_p_2": [-89.708, -0.005],
        "relative_v_2": [-10.132, 13.686],
    }

    scenarioInstance.set_save_data(True)  # Save output data
    scenarioInstance.set_save_video(False)  # Save simulator video
    scenarioInstance.set_render(True)  # Render simulator
    scenarioInstance.set_debug(False)  # Print debug

    scenarioInstance.setup(
        params_dict, loss_weight=[w, 1 - w, 50], ttc_mode="longitudinal"
    )

    scenarioInstance.search_spso(n_particles=n_particles, max_iter=max_iter)

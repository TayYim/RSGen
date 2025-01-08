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


class FrontBrake(BasicTrafficEnv):


    def _process_natural_data(self):
        """
        Process natural data
        """
        with open(self.natural_data_path, "r") as infile:
            sr = infile.read()
            hb_vehicles = json.loads(sr)
        decision_var = []
        for hb_vehicle in hb_vehicles:
            range = hb_vehicle["start_info"]["range"] * 0.3048
            range_rate = hb_vehicle["start_info"]["range_rate"] * 0.3048
            speed = hb_vehicle["start_info"]["ego"]["speed"] * 0.3048
            decision_var.append([range, range_rate, speed])

        # Save result
        self.decision_var = np.array(decision_var)

    # Override in subclass. Order sensitive

    def _calculate_similarity(self):
        inputs = [
            self.search_collector["relative_p"][-1],
            self.search_collector["relative_v"][-1],
            self.search_collector["absolute_v"][-1],
        ]

        similarity = super()._calculate_similarity(inputs)

        return similarity

    def run_step(self, absolute_v, relative_p, relative_v):

        self.env.unwrapped.config["absolute_v"] = absolute_v
        self.env.unwrapped.config["relative_p"] = relative_p
        self.env.unwrapped.config["relative_v"] = relative_v

        self.search_collector["absolute_v"].append(absolute_v)
        self.search_collector["relative_v"].append(relative_v)
        self.search_collector["relative_p"].append(relative_p)

        return super().run_step()

    @BasicTrafficEnv.search_method_decorator("natural")
    def search_natural(self):

        total_len = len(self.decision_var)

        for i in range(total_len):
            absolute_v = self.decision_var[i][2]
            relative_p = self.decision_var[i][0]
            relative_v = self.decision_var[i][1]
            if (
                self.within_bounds("absolute_v", absolute_v)
                and self.within_bounds("relative_p", relative_p)
                and self.within_bounds("relative_v", relative_v)
            ):
                self.run_step(absolute_v, relative_p, relative_v)


if __name__ == "__main__":

    # -- Init params --
    name = "front_brake"
    w = 0.7
    n_particles = 1
    max_iter = 1
    agent = "apollo"
    random_seed = 0
    formatted_time = datetime.now().strftime("%m%d%H%M")
    checkpoint = False

    output_path = Path(
        f"output-demo0108/{name}/{agent}/{formatted_time}_{w}_{n_particles}x{max_iter}_{random_seed}"
    )

    ## Checkpoint
    # max_iter = 5
    # checkpoint = True
    # output_path = Path("output-checkpoint/front_brake/tfpp/04012148_0.0_20x50_0")

    # Init env
    # debug=True will print step info
    scenarioInstance = FrontBrake(
        env_name="front-brake-v0",
        render=False,
        model_path=Path("data/models/MAF_hb_trk_best.pth"),
        natural_data_path=Path("data/datasets/hb_trk.json"),
        output_path=output_path,
        name=name,
        debug=False,
        random_seed=random_seed,
        agent=agent,
        checkpoint=checkpoint,
    )

    # Order of parameters should be consistent with the order of parameters in run_step()
    params_dict = {
        "absolute_v": [0, 26.368],
        "relative_p": [0.114, 59.384],
        "relative_v": [-16.993, 7.593],
    }

    scenarioInstance.set_save_data(True)  # Save output data
    scenarioInstance.set_save_video(False)  # Save simulator video
    scenarioInstance.set_render(False)  # Render simulator
    scenarioInstance.set_debug(False)  # Print debug

    scenarioInstance.setup(
        params_dict, loss_weight=[w, 1 - w, 50], ttc_mode="longitudinal"
    )

    scenarioInstance.search_spso(n_particles=n_particles, max_iter=max_iter)
    # scenarioInstance.search_ga(n_population=n_particles, n_generation=max_iter)

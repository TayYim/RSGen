import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium.wrappers import RecordVideo
from sko.PSO import PSO
from sko.GA import GA
from bayes_opt import BayesianOptimization
from spso import SPSO
from torch.distributions import MultivariateNormal
from scripts.flow_model import NormalizingFlowModel, MAF
import torch
import time
from datetime import datetime
import os
import csv
import random
import functools
import subprocess
import json
import pickle
import os
import psutil
import signal
import sys
import math
from env_utils import (
    cal_angle,
    calculate_min_ttc,
    calculate_min_thw,
    show_simulation_agents_data,
    change_route_value,
)

np.seterr(divide="ignore", invalid="ignore")


class BasicTrafficEnv:
    def __init__(
        self,
        env_name,
        name="BasicTrafficEnv",
        render=False,
        save_video=False,
        save_data=False,
        model_path=None,
        natural_data_path=None,
        output_path="output/",
        debug=False,
        random_seed=0,
        agent="idm",
        checkpoint=None,
    ):
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.render = render
        self.save_video = save_video
        self.search_method = None
        self.generative_model = None
        self.debug = debug  # If True, will output status in console
        self.params_dict = None  # Param dict, need to be initialized
        self.loss_weight = [0.5, 0.5, 50]  # Weight of loss， [ttc, similarity, collision]
        self.save_data = save_data  # Save output data

        self.set_random_seed(random_seed)

        # Vars for natural data
        self.decision_var = None
        self.x_mean = None
        self.x_std = None

        # Vars for path
        self.output_root_path = output_path
        self.model_path = model_path
        self.natural_data_path = natural_data_path
        self.set_name(name)

        # Setting for TTC calculation
        self.ttc_mode = "longitudinal"  # "longitudinal" or "cross"

        # AV model:"AV-I" or "AV-II-CROSS"
        self.av_model_type = "AV-I"
        self.av_model = None

        # Extra info, may be used for output, as extra flag info
        self.extra_info = None

        self.checkpoint = checkpoint  # Whether to use checkpoint

        # Agent Config
        self.agent = agent  # idm, ba, interfuser, apollo, tfpp
        self.simulator = "highway-env"  # carla, highway-env
        current_path = os.path.dirname(os.path.abspath(__file__))
        parent_path = os.path.dirname(current_path)

        # Read config json
        with open(os.path.join(parent_path, "config.json"), "r") as f:
            self.config = json.load(f)

        simulate_script_dict = {
            "ba": os.path.join(parent_path, "carla_simulate/simulate_ba.sh"),
            "interfuser": os.path.join(
                parent_path, "carla_simulate/simulate_interfuser.sh"
            ),
            "tfpp": os.path.join(parent_path, "carla_simulate/simulate_tfpp.sh"),
            "apollo": os.path.join(parent_path, "carla_simulate/simulate_apollo.sh"),
        }
        if self.agent in simulate_script_dict:
            self.simulate_script = simulate_script_dict[self.agent]
            self.simulator = "carla"

        # Carla config
        self.route_file = os.path.join(
            self.config["env"]["LEADERBOARD_ROOT"], "data/routes_osg.xml"
        )
        this_scenario_config = None
        for scenario in self.config["scenarios"]:
            if scenario["name"] == name:
                this_scenario_config = scenario
                break
        self.route_id = this_scenario_config["route_id"]
        self.carla_shell = os.path.join(self.config["env"]["CARLA_ROOT"], "CarlaUE4.sh")
        if self.agent == "apollo":
            self.carla_restart_gap = 1
        else:
            self.carla_restart_gap = 10

    # setup the necessary parameters, load the model

    def setup(
        self,
        params_dict,
        loss_weight=[0.5, 0.5, 50],
        ttc_mode="longitudinal",
        av_model_type="AV-I",
        av_model=None,
    ):
        self.set_params_dict(params_dict)
        self.set_loss_weight(loss_weight)

        if self.model_path is not None:
            self._init_generative_model()

        if self.natural_data_path is not None:
            self._process_natural_data()
            # Calculate the statistics of the natural data
            self.x_std = np.std(self.decision_var, axis=0)
            self.x_mean = np.mean(self.decision_var, axis=0)
            self._describe_natural_data()

        # init data collector
        self._clear_data()

        self.ttc_mode = ttc_mode

        self.av_model_type = av_model_type
        self.av_model = av_model
        self._process_av_model()

    def set_av_model(self, av_model):
        self.av_model = av_model

    def set_av_model_type(self, av_model_type):
        self.av_model_type = av_model_type

    def _process_av_model(self):
        # Setting for AV model
        if self.av_model_type == "AV-I":
            pass
        elif self.av_model_type == "AV-II-CROSS":
            self.env.configure(
                {
                    "vehicles_count": 10,
                    "observation": {
                        "type": "Kinematics",
                        "vehicles_count": 5,
                        "absolute": False,
                        "features": [
                            "presence",
                            "x",
                            "y",
                            "vx",
                            "vy",
                            "cos_h",
                            "sin_h",
                        ],
                    },
                    "policy_frequency": 15,
                    "high_speed_reward": 1,
                    "v_ego": 8,
                    "r_ego": 20,
                    "v_1": 8,
                    "r_1": 20,
                }
            )
        else:
            raise ValueError("Wrong AV model type!")

    def _describe_natural_data(self):
        """Describe the natural data"""
        data = self.decision_var

        n = len(data)

        x_max = np.max(data, axis=0)
        x_min = np.min(data, axis=0)
        x_max = [round(x, 3) for x in x_max.tolist()]
        x_min = [round(x, 3) for x in x_min.tolist()]

        # print info above
        print("=== Natural Data ===")
        print("Amount: ", n)
        print("Mean: ", self.x_mean)
        print("Std: ", self.x_std)
        print("Min: ", x_min)
        print("Max: ", x_max)

    # setter of params_dict

    def set_params_dict(self, params_dict):
        self.params_dict = params_dict
        self.dim = len(params_dict)

    # setter of loss_weight
    def set_loss_weight(self, loss_weight):
        self.loss_weight = loss_weight

    # setter of save_data

    def set_save_data(self, save_data):
        self.save_data = save_data

    # setter of render
    def set_render(self, render):
        self.render = render

    # setter of debug
    def set_debug(self, debug):
        self.debug = debug

    # setter of loss_weight
    def set_loss_weight(self, loss_weight):
        self.loss_weight = loss_weight

    # setter of name
    def set_name(self, name):
        self.name = name
        # self.output_path = os.path.join(self.output_root_path, name)
        self.output_path = self.output_root_path  # Remove the name for OSG framework
        self.video_path = os.path.join(self.output_root_path, "videos", self.name)

    # setter of random_seed
    def set_random_seed(self, random_seed):
        self.random_seed = random_seed
        np.random.seed(self.random_seed)  # For sko search methods
        random.seed(self.random_seed)

    # settet of save_video
    def set_save_video(self, save_video):
        self.save_video = save_video

    def set_extra_info(self, extra_info):
        self.extra_info = extra_info

    def _init_generative_model(self):
        # Init the generative model
        flow = MAF
        flows = [flow(dim=self.dim) for _ in range(self.dim)]

        prior = MultivariateNormal(torch.zeros(self.dim), torch.eye(self.dim))
        self.generative_model = NormalizingFlowModel(prior, flows)
        self.generative_model.eval()
        checkpoint = torch.load(self.model_path)
        self.generative_model.load_state_dict(checkpoint["net"])

    # Need to be overrided in subclass
    # The order and dimension of the parameters should be consistent with the input of the generative model
    def _process_natural_data(self):
        """
        Process the natural data, get the mean and std
        """
        # Read the natural data and store it in self.decision_var
        # Calculate the statistics of the natural data
        pass

    def _clear_data(self):
        """
        Init the data collector
        """
        if not self.checkpoint:
            self.statistic_collector = {
                "ttc": [],
                "thw": [],
                "collision": [],
                "search_id": [],
                "label": [],
                "obs": [],
            }
            # distance_x: ego vehicle's x direction mileage
            # TODO change to distance, not adapted yet
            self.search_collector = {
                "search_id": [],
                "ttc": [],
                "similarity": [],
                "loss": [],
                "time": [],
                "distance": [],
                "collision_status": [],
            }
            for key in self.params_dict.keys():
                self.search_collector[key] = []
        else:
            search_collector_path = os.path.join(
                self.output_path, "search_collector.pkl"
            )
            if os.path.exists(search_collector_path):
                with open(search_collector_path, "rb") as file:
                    self.search_collector = pickle.load(file)
            else:
                raise ValueError("No search_collector.pkl found")

            statistic_collector_path = os.path.join(
                self.output_path, "statistic_collector.pkl"
            )
            if os.path.exists(statistic_collector_path):
                with open(statistic_collector_path, "rb") as file:
                    self.statistic_collector = pickle.load(file)
            else:
                raise ValueError("No statistic_collector.pkl found")

    def _normalize_input(self, x):
        """
        Standardize the input
        """
        return (np.array(x) - self.x_mean) / self.x_std

    def _collect_data_step(self, obs, label, search_id, collision=False):
        """
        Collect the statistic data in each step of the simulation
        Override this method if you need more statistic data
        """
        if collision:
            self.statistic_collector["ttc"].append(0)
            self.statistic_collector["thw"].append(0)
            self.statistic_collector["collision"].append(1)
        else:
            ttc = calculate_min_ttc(obs, self.ttc_mode)
            self.statistic_collector["ttc"].append(ttc)

            thw = calculate_min_thw(obs, self.ttc_mode)
            self.statistic_collector["thw"].append(thw)

            self.statistic_collector["collision"].append(0)

        self.statistic_collector["search_id"].append(search_id)
        self.statistic_collector["obs"].append(obs)
        self.statistic_collector["label"].append(label)

    # Need to be overrided in subclass
    # Ajdust the order and dimension of the parameters according to the model requirements
    def run_step(self):
        # self.env.unwrapped.config["absolute_v"] = absolute_v
        # self.env.unwrapped.config["relative_p"] = relative_p
        # self.env.unwrapped.config["relative_v"] = relative_v

        # self.search_collector["absolute_v"].append(absolute_v)
        # self.search_collector["relative_v"].append(relative_v)
        # self.search_collector["relative_p"].append(relative_p)

        # Call _simulate to simulate
        if self.simulator == "highway-env":
            return self._simulate()
        elif self.simulator == "carla":
            return self._simulate_carla()

    # Wrap the run_step, input is a list, for sko to use
    def _run_step_warp(self, x):
        if self.search_method == "spso":
            return (self.run_step(*x),)
        else:
            return self.run_step(*x)

    # Check if the parameters are within the reasonable range, according to params_dict
    def within_bounds(self, var_name, var_value):
        if (
            var_value >= self.params_dict[var_name][0]
            and var_value <= self.params_dict[var_name][1]
        ):
            return True
        else:
            return False

    # Need to be overrided in subclass, the order and dimension of the parameters should be consistent with the input of the generative model
    def _calculate_similarity(self, input=None):
        """
        Calculate the similarity of the current parameters, return a likelihood value, the larger the value, the more natural
        """
        # Order of parameters: relative distance, relative velocity, absolute velocity
        # Output of the model: three values, the sum of the last two is the likelihood, the larger the better
        inputs = input
        try:
            inputs = self._normalize_input(inputs)
            output_tensor = self.generative_model(torch.Tensor([list(inputs)]))
            similarity = float(output_tensor[1] + output_tensor[2])
            similarity = max(similarity, -100)  # Restrict the minimum value
        except Exception as e:
            if self.debug:
                print("Failed to calculate similarity: {}".format(e))
            similarity = 0

        return similarity

    def _simulate(self):
        """
        Run a simulation according to the set env
        """

        # generate a search_id with timestamp and a random number
        search_id = str(time.time()) + "-" + str(random.randint(0, 1000))

        if self.save_video:
            # make self.env.unwrapped.config a string
            if self.extra_info is None:
                config_str = search_id + "_"
            else:
                config_str = str(self.extra_info) + "_" + search_id + "_"
            for key in self.env.unwrapped.config.keys():
                # if key in self.params_dict's keys
                if key in self.params_dict.keys():
                    config_str += "{}_{}_".format(
                        key, round(self.env.unwrapped.config[key], 3)
                    )
            this_video_directory = os.path.join(self.video_path, config_str)
            self.env = RecordVideo(
                self.env,
                video_folder=this_video_directory,
                episode_trigger=lambda e: True,
            )
            self.env.unwrapped.set_record_video_wrapper(self.env)

        obs = self.env.reset()
        ttcs = []
        distance_x = 0
        start_position_x = 0
        last_position_x = 0
        collision_flag = False
        agents_datas = []

        # TODO refactor the distance calculation
        distance = 0
        curr_position = (0, 0)
        last_position = (0, 0)
        ego_vehicle_data = None

        for i in range(15):
            # use av_model if it's not None
            if self.av_model is not None:
                action, _ = self.av_model.predict(obs)
                obs, reward, done, truncated, info = self.env.step(action)
            else:
                obs, reward, done, truncated, info = self.env.step(None)

            if self.render:
                self.env.render()

            if len(obs) < 2:  # no other vehicles detected
                print("no other vehicles detected")
                break

            if np.all(obs[0] == 0):  # ego vehicle not detected
                break

            # append every item in info['agents_data'] to agents_datas
            for agent_data in info["agents_data"]:
                agents_datas.append(agent_data)
                if agent_data["is_ego"]:
                    ego_vehicle_data = agent_data
            if ego_vehicle_data is None:
                raise ValueError("Ego is None")

            curr_position = (ego_vehicle_data["x"], ego_vehicle_data["y"])

            if i == 0:
                start_position_x = obs[0][1]
                last_position = (ego_vehicle_data["x"], ego_vehicle_data["y"])

            # calculate distance
            distance += np.sqrt(
                (curr_position[0] - last_position[0]) ** 2
                + (curr_position[1] - last_position[1]) ** 2
            )
            last_position = curr_position

            # check collision
            for vehicle in self.env.env.unwrapped.road.vehicles:
                if vehicle.is_ego and vehicle.crashed:
                    collision_flag = True

            # Collect data
            if collision_flag:
                label = 0  
                # TODO consider the distinction between the collision of the main vehicle and the collision of other vehicles. Currently only the collision of the main vehicle is considered
                ttcs.append(0)
            else:
                label = 2
                ttc = calculate_min_ttc(obs, self.ttc_mode)
                if ttc is not None:
                    ttcs.append(ttc)
            self._collect_data_step(obs, label, search_id, collision=collision_flag)

            if done:
                break

        last_position_x = obs[0][1]
        # distance_x = last_position_x - start_position_x
        # TODO use the distance of the ego vehicle
        distance_x = distance

        self.env.close()

        similarity = self._calculate_similarity()

        # Calculate the loss
        if len(ttcs) == 0:  # If there is no ttc, set it to 10
            ttcs = [10]
        min_ttc = min(ttcs) # Positive number, minimum ttc

        if collision_flag:
            collision_loss = 1

            # If you need to enable the collision rate as the loss collision item, uncomment the following two lines
            # collision_loss = 1/distance if distance > 0 else 0 # tmp try
            # collision_loss = min(1, collision_loss)
        else:
            collision_loss = 0

        # Old, for demo
        # if self.search_method == "bys":
        #     loss = (
        #         -1 * min_ttc + collision_loss * self.loss_weight[2]
        #     ) * self.loss_weight[0] + similarity * self.loss_weight[1]
        # else:
        #     loss = (min_ttc - collision_loss * self.loss_weight[2]) * self.loss_weight[
        #         0
        #     ] - similarity * self.loss_weight[1]

        #less better
        w = self.loss_weight[0]
        A = min_ttc - collision_loss * 50
        A = (A - -50) / (10 - -50)  # normalize 
        N = (similarity - -100) / (0 - -100)  # normalize 
        N = 1 - N
        loss = (A ** (w**2) + N ** ((1 - w) ** 2)) ** math.exp(w * (1 - w))

        # Collect search data
        self.search_collector["ttc"].append(min_ttc)
        self.search_collector["similarity"].append(similarity)
        self.search_collector["loss"].append(loss)
        self.search_collector["distance"].append(distance_x)
        self.search_collector["search_id"].append(search_id)

        # save agent data with the save video
        if self.save_video:
            # agents_datas is a list of dict, make it a dataframe
            agents_datas_df = pd.DataFrame(agents_datas)

            show_simulation_agents_data(agents_datas_df, this_video_directory)

        if self.debug:
            print("===============================")
            for key in self.search_collector.keys():
                if key == "time":
                    continue
                print("{}: {:.2f}".format(key, self.search_collector[key][-1]))

        return loss

    def _simulate_carla(self, attempt=1):

        search_id = str(time.time()) + "-" + str(random.randint(0, 1000))

        # If reach the max attempt, return the previous loss
        if attempt <= 5:

            # Use Carla leaderboard to simulate
            # TODO: use self.
            search_count = len(self.search_collector["search_id"])

            print("search_count: ", search_count)

            # 0. Start Carla
            ## Restart Carla every self.carla_restart_gap simulations
            # TODO: if is the end, kill the process
            if search_count % self.carla_restart_gap == 0:
                self._restart_carla()
            while not self._is_carla_running():
                self._restart_carla()

            # 2. Set params
            for key in self.params_dict.keys():
                change_route_value(
                    self.route_file, self.route_id, key, self.search_collector[key][-1]
                )

            # 3. Start simulation
            process = subprocess.Popen(
                ["/bin/bash", self.simulate_script, self.route_id],
                stdout=subprocess.PIPE,
            )

            try:
                process.wait(timeout=300)  # 5 mins
            except:
                print(f"Simulation timed out at {datetime.now().strftime('%m%d%H%M')}")
                process.kill()  # End the process
                if self.agent == "apollo":
                    self._restart_apollo()
                self._restart_carla()
                print(f"Process terminated. Retrying... [No.{attempt}]")
                return self._simulate_carla(attempt=attempt + 1)  # Recursion call
        else:
            print("Max attempt reached. Return previous loss.")
            self._kill_carla()
            if self.agent == "apollo":
                self._restart_apollo()

        # 4. Get result
        # Read the result from the epoch_result.json
        with open("epoch_result.json", "r") as f:
            result = json.load(f)

        collision_flag = result["collision_flag"]
        min_ttc = result["min_ttc"]
        collision_status = result["collision_status"]
        distance = result["distance"]
        similarity = self._calculate_similarity()

        loss = (min_ttc - collision_flag * self.loss_weight[2]) * self.loss_weight[
            0
        ] - similarity * self.loss_weight[1]

        # Collect search data
        self.search_collector["ttc"].append(min_ttc)
        self.search_collector["similarity"].append(similarity)
        self.search_collector["loss"].append(loss)
        self.search_collector["search_id"].append(search_id)
        self.search_collector["collision_status"].append(collision_status)
        self.search_collector["distance"].append(distance)

        # 5. Return loss

        return loss

    # decorator of search_ methods
    # Decorator of search_ methods, complete some common operations
    def search_method_decorator(method_name):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(self, *args, **kwargs):
                self.search_method = method_name
                self._clear_data()
                start_time = time.perf_counter()

                if self.debug:
                    print("Start to search by {}...".format(method_name))

                # reset random seed
                # Considering the fork feature of multiple processes, the random seed needs to be set again here, otherwise the same random number will appear
                self.set_random_seed(self.random_seed)

                def terminate(start_time):
                    end_time = time.perf_counter()
                    time_usage = end_time - start_time
                    self.search_collector["time"].append(time_usage)

                    self._save_pickle(
                        self.output_path, "search_collector.pkl", self.search_collector
                    )
                    self._save_pickle(
                        self.output_path,
                        "statistic_collector.pkl",
                        self.statistic_collector,
                    )

                    if self.save_data:
                        self._save_csv(
                            self.output_path,
                            "statistic_{}.csv".format(self.search_method),
                            data=self.statistic_collector,
                        )
                        self._save_csv(
                            self.output_path,
                            "search_{}.csv".format(self.search_method),
                            data=self.search_collector,
                        )

                    print(
                        "Method:{},time:{:.2f}s".format(self.search_method, time_usage)
                    )
                    print(f"Restuls saved in {self.output_path}")

                    if self.simulator == "carla":
                        self._kill_carla()

                def signal_handler(sig, frame):
                    print("User interrupt the process")
                    terminate(start_time)
                    sys.exit(0)

                signal.signal(signal.SIGINT, signal_handler)

                retval = func(self, *args, **kwargs) # Execute the search method

                terminate(start_time)

                return retval

            return wrapper

        return decorator

    @search_method_decorator("bys")
    def search_bys(self, init_points=0, n_iter=50):
        # Bayesian Optimization

        # Define search space
        pbounds = self.params_dict
        optimizer = BayesianOptimization(
            f=self.run_step, pbounds=pbounds, random_state=self.random_seed, verbose=0
        )

        # if save:
        #     file_path = os.path.join(self.output_path, self.name + "_bys.json")
        #     logger = JSONLogger(path=file_path)
        #     optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

        optimizer.maximize(init_points=init_points, n_iter=n_iter)

    @search_method_decorator("pso")
    def search_pso(self, n_particles=40, max_iter=50, w=0.8, c1=0.5, c2=0.5):
        # w: inertia weight
        # c1: individual memory
        # c2: collective memory
        # Particle Swarm Optimization

        # Define search space
        x_lb = [x[0] for x in self.params_dict.values()]
        x_ub = [x[1] for x in self.params_dict.values()]
        dim = len(x_lb)
        pso = PSO(
            func=self._run_step_warp,
            dim=dim,
            pop=n_particles,
            max_iter=max_iter,
            lb=x_lb,
            ub=x_ub,
            w=w,
            c1=c1,
            c2=c2,
        )
        pso.run()

    @search_method_decorator("spso")
    def search_spso(self, n_particles=40, max_iter=50, w=0.8, c1=0.5, c2=0.5):
        # w: inertia weight
        # c1: individual memory
        # c2: collective memory
        # Particle Swarm Optimization

        # Define search space
        x_lb = [x[0] for x in self.params_dict.values()]
        x_ub = [x[1] for x in self.params_dict.values()]
        dim = len(x_lb)
        spso = SPSO(
            func=self._run_step_warp,
            dim=dim,
            pop=n_particles,
            max_iter=max_iter,
            lb=x_lb,
            ub=x_ub,
            w=w,
            c1=c1,
            c2=c2,
            output_path=self.output_path,
        )
        if self.checkpoint:
            checkpoint = os.path.join(self.output_path, "checkpoint.pkl")
            species, logbook = spso.run(checkpoint=checkpoint)
        else:
            species, logbook = spso.run()
        print("=====Show the species=====")
        print(len(species))

        # use pickle to save species and logbook to self.output_path
        self._save_pickle(self.output_path, "species.pkl", species)
        self._save_pickle(self.output_path, "logbook.pkl", logbook)

    @search_method_decorator("replay")
    def replay(self, params_list):
        # params_list: should contain a list of parameters' list

        for this_params in params_list:
            self._run_step_warp(this_params)

    @search_method_decorator("random")
    def search_random(self, n_iter=50):
        # Random Search

        # Define search space
        x_lb = [x[0] for x in self.params_dict.values()]
        x_ub = [x[1] for x in self.params_dict.values()]
        dim = len(x_lb)

        for i in range(n_iter):
            if self.debug:
                print("Step: {}".format(i))
            x = np.random.uniform(x_lb, x_ub, dim)
            self._run_step_warp(x)

    @search_method_decorator("ga")
    def search_ga(self, n_population=50, n_generation=50, prob_mut=0.01):
        # Genetic Algorithm

        # Define search space
        x_lb = [x[0] for x in self.params_dict.values()]
        x_ub = [x[1] for x in self.params_dict.values()]
        dim = len(x_lb)

        ga = GA(
            func=self._run_step_warp,
            n_dim=dim,
            size_pop=n_population,
            max_iter=n_generation,
            lb=x_lb,
            ub=x_ub,
            prob_mut=prob_mut,
        )
        ga.run()

    # Override as needed, may vary due to different natural data formats

    @search_method_decorator("natural")
    def search_natural(self):
        # Simulate based on natural data

        total_len = len(self.decision_var)

        if self.debug:
            print("Total data: {}".format(total_len))

        for i in range(total_len):
            if self.debug:
                print("Step: {}".format(i))
            absolute_v = random.randint(15, 30)
            self.run_step(absolute_v, self.decision_var[i][0], self.decision_var[i][1])

    def _save_csv(self, path, filename, data):
        """
        save dict data to csv
        """
        file_path = os.path.join(path, filename)
        if not os.path.exists(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            header = data.keys()
            writer = csv.writer(file)
            writer.writerow(header)
            length = len(data[list(header)[0]])
            for i in range(length):
                row = []
                for item in header:
                    if len(data[item]) <= i:
                        row.append("")
                    try:
                        row.append(data[item][i])
                    except:
                        pass
                writer.writerow(row)

    def _save_pickle(self, path, filename, data):
        """
        save dict data to csv
        """
        file_path = os.path.join(path, filename)
        if not os.path.exists(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file:
            pickle.dump(data, file)

    def _kill_carla(self):
        # Run multiple times to ensure the process is killed
        # Also kill leaderboard to avoid zombie process
        subprocess.call(["pkill", "-f", "leaderboard_evaluator"])
        subprocess.call(["pkill", "-f", "leaderboard_evaluator"])
        subprocess.call(["pkill", "-f", "leaderboard_evaluator"])
        subprocess.call(["pkill", "CarlaUE4"])
        subprocess.call(["pkill", "CarlaUE4"])
        subprocess.call(["pkill", "CarlaUE4"])
        if self.agent == "interfuser":
            subprocess.call(["pkill", "-f", "interfuser"])
        if self.agent == "tfpp":
            subprocess.call(["pkill", "-f", "tfpp"])
        if self.agent == "ba":
            subprocess.call(["pkill", "-f", "simulate_ba"])
        time.sleep(10)

    def _restart_apollo(self):
        # Restart docker
        cmd = "docker ps -aqf 'name=apollo_dev_' | xargs -r docker restart"
        _ = subprocess.call(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        time.sleep(1)

    def _restart_carla(self):
        self._kill_carla()
        # cmds = [self.carla_shell, "-ResX=400", "-ResY=300"]
        cmds = [self.carla_shell, "-ResX=600", "-ResY=600"]
        if not self.render:
            cmds.append("-RenderOffScreen")
        env = os.environ.copy()
        # Check for Ubuntu server
        if "DISPLAY" not in env.keys():
            env["DISPLAY"] = ":10.0"
        self.process = subprocess.Popen(
            cmds, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env
        )
        time.sleep(10)

    def _is_process_running(self, process_name):
        # Enumerate all running processes
        for proc in psutil.process_iter(attrs=["name"]):
            try:
                # Find the process name that matches the specified name
                if process_name.lower() in proc.info["name"].lower():
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        return False

    def _is_carla_running(self):
        return self._is_process_running("CarlaUE4")

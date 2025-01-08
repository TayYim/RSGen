import numpy as np
from matplotlib import pyplot as plt
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from multiprocess import Pool

import copy
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import os
from stable_baselines3 import PPO

from scripts.wandb_run_provider import WanDBRunProvider
import wandb
import xml.etree.ElementTree as ET


def cal_angle(v1, v2):
    dx1, dy1 = v1
    dx2, dy2 = v2
    angle1 = np.arctan2(dy1, dx1)
    angle1 = int(angle1 * 180 / np.pi)
    angle2 = np.arctan2(dy2, dx2)
    angle2 = int(angle2 * 180 / np.pi)
    if angle1 * angle2 >= 0:
        included_angle = abs(angle1 - angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
        if included_angle > 180:
            included_angle = 360 - included_angle
    return included_angle


def calculate_min_ttc(obs, mode="longitudinal"):
    """
    Calculate the minimum TTC at the current moment.

    mode: "longitudinal" or "cross"
    """
    MAX_TTC = 10
    ttc_list = []
    for i in range(1, len(obs)):
        # If the agent does not exist
        if np.all(obs[i] == 0):
            continue

        if mode == "longitudinal":
            vector_p = np.array([obs[0, 1] - obs[i, 1], obs[0, 2] - obs[i, 2]])
            vector_v = np.array([obs[0, 3] - obs[i, 3], obs[0, 4] - obs[i, 4]])
            angle = cal_angle(vector_p, vector_v)
            v = np.sqrt(vector_v[0] ** 2 + vector_v[1] ** 2)
            v_projection = v * np.cos(angle / 180 * np.pi)
            if v_projection < 0:
                # ttc = np.sqrt(vector_p[0] ** 2 + vector_p[1] ** 2) / v_projection
                ttc = vector_p[0] / vector_v[0] 

                if ttc < 0:
                    ttc_list.append(abs(ttc))

        # TODO: refactoring, simplify the code, and consider the edge case
        if mode == "cross":
            p_0 = (obs[0, 1], obs[0, 2])
            v_0 = (obs[0, 3], obs[0, 4])
            # count the angle of vo and the x-axis
            # The value range of the result of atan2 is from -180 to 180 degrees
            th_0 = np.arctan2(v_0[1], v_0[0])
            for i in range(1, len(obs)):
                p_i = (obs[i, 1], obs[i, 2])
                v_i = (obs[i, 3], obs[i, 4])
                th_i = np.arctan2(v_i[1], v_i[0])

                # Calculate determinant of the matrix formed by the direction vectors
                det = v_0[0] * v_i[1] - v_i[0] * v_0[1]
                if det == 0:
                    # Vectors are parallel, there is no intersection
                    # Adopt the TTC algorithm in the case of parallelism
                    vector_p = np.array([obs[0, 1] - obs[i, 1], obs[0, 2] - obs[i, 2]])
                    vector_v = np.array([obs[0, 3] - obs[i, 3], obs[0, 4] - obs[i, 4]])
                    angle = cal_angle(vector_p, vector_v)
                    v = np.sqrt(vector_v[0] ** 2 + vector_v[1] ** 2)
                    v_projection = v * np.cos(angle / 180 * np.pi)
                    if v_projection < 0:
                        ttc = (
                            np.sqrt(vector_p[0] ** 2 + vector_p[1] ** 2) / v_projection
                        )
                        ttc_list.append(abs(ttc))
                else:
                    # Calculate the parameter values for each vector
                    t1 = (v_i[0] * (p_0[1] - p_i[1]) - v_i[1] * (p_0[0] - p_i[0])) / det
                    t2 = (v_0[0] * (p_0[1] - p_i[1]) - v_0[1] * (p_0[0] - p_i[0])) / det

                    # Calculate the intersection point
                    x_cross = p_0[0] + v_0[0] * t1
                    y_cross = p_0[1] + v_0[1] * t1

                    p_c = (x_cross, y_cross)

                    TTX_0 = (
                        np.sqrt((p_c[0] - p_0[0]) ** 2 + (p_c[1] - p_0[1]) ** 2)
                        / np.sqrt(v_0[0] ** 2 + v_0[1] ** 2)
                        * np.sign(
                            (p_c[0] - p_0[0]) * v_0[0] + (p_c[1] - p_0[1]) * v_0[1]
                        )
                    )
                    TTX_i = (
                        np.sqrt((p_c[0] - p_i[0]) ** 2 + (p_c[1] - p_i[1]) ** 2)
                        / np.sqrt(v_i[0] ** 2 + v_i[1] ** 2)
                        * np.sign(
                            (p_c[0] - p_i[0]) * v_i[0] + (p_c[1] - p_i[1]) * v_i[1]
                        )
                    )

                    # Threshold is the length of the vehicle body divided by the maximum speed
                    delta_threshold = 5 / max(
                        np.sqrt(v_0[0] ** 2 + v_0[1] ** 2),
                        np.sqrt(v_i[0] ** 2 + v_i[1] ** 2),
                    )

                    if TTX_0 > 0 and TTX_i > 0:
                        if abs(TTX_0 - TTX_i) < delta_threshold:
                            ttc_list.append(TTX_0)

    if len(ttc_list) == 0:
        return None
    elif min(ttc_list) > MAX_TTC: 
        return None
    else:
        return min(ttc_list)


def calculate_min_thw(obs, mode="longitudinal"):
    MAX_THW = 10
    thw_list = []
    for i in range(1, len(obs)):
        if np.all(obs[i] == 0):
            continue

        if mode == "longitudinal":
            vector_p = np.array([obs[0, 1] - obs[i, 1], obs[0, 2] - obs[i, 2]])
            vector_v = np.array([obs[0, 3], obs[0, 4]])  # Only need to consider the speed of the ego vehicle
            angle = cal_angle(vector_p, vector_v)
            v = np.sqrt(vector_v[0] ** 2 + vector_v[1] ** 2)
            v_projection = v * np.cos(angle / 180 * np.pi)
            if v_projection < 0:
                # thw = np.sqrt(vector_p[0] ** 2 + vector_p[1] ** 2) / v_projection

                thw = vector_p[0] / vector_v[0]
                if thw < 0:
                    thw_list.append(abs(thw))

        # Similar to TTC
        if mode == "cross":
            p_0 = (obs[0, 1], obs[0, 2])
            v_0 = (obs[0, 3], obs[0, 4])
            # count the angle of vo and the x-axis
            th_0 = np.arctan2(v_0[1], v_0[0])
            for i in range(1, len(obs)):
                p_i = (obs[i, 1], obs[i, 2])
                v_i = (obs[i, 3], obs[i, 4])
                th_i = np.arctan2(v_i[1], v_i[0])

                # Calculate determinant of the matrix formed by the direction vectors
                det = v_0[0] * v_i[1] - v_i[0] * v_0[1]
                if det == 0:
                    # Vectors are parallel, there is no intersection
                    vector_p = np.array([obs[0, 1] - obs[i, 1], obs[0, 2] - obs[i, 2]])
                    vector_v = np.array([obs[0, 3], obs[0, 4]])
                    angle = cal_angle(vector_p, vector_v)
                    v = np.sqrt(vector_v[0] ** 2 + vector_v[1] ** 2)
                    v_projection = v * np.cos(angle / 180 * np.pi)
                    if v_projection < 0:
                        thw = (
                            np.sqrt(vector_p[0] ** 2 + vector_p[1] ** 2) / v_projection
                        )
                        thw_list.append(abs(thw))
                else:
                    # Calculate the parameter values for each vector
                    t1 = (v_i[0] * (p_0[1] - p_i[1]) - v_i[1] * (p_0[0] - p_i[0])) / det
                    t2 = (v_0[0] * (p_0[1] - p_i[1]) - v_0[1] * (p_0[0] - p_i[0])) / det

                    # Calculate the intersection point
                    x_cross = p_0[0] + v_0[0] * t1
                    y_cross = p_0[1] + v_0[1] * t1

                    p_c = (x_cross, y_cross)

                    dis_to_x_0 = np.sqrt(
                        (p_c[0] - p_0[0]) ** 2 + (p_c[1] - p_0[1]) ** 2
                    )
                    v_project_0 = np.sqrt(v_0[0] ** 2 + v_0[1] ** 2) * np.sign(
                        (p_c[0] - p_0[0]) * v_0[0] + (p_c[1] - p_0[1]) * v_0[1]
                    )

                    dis_to_x_i = np.sqrt(
                        (p_c[0] - p_i[0]) ** 2 + (p_c[1] - p_i[1]) ** 2
                    )
                    v_project_i = np.sqrt(v_i[0] ** 2 + v_i[1] ** 2) * np.sign(
                        (p_c[0] - p_i[0]) * v_i[0] + (p_c[1] - p_i[1]) * v_i[1]
                    )

                    TTX_0 = dis_to_x_0 / v_project_0
                    TTX_i = dis_to_x_i / v_project_i

                    # If the distance is close enough, calculate the thw
                    if max(TTX_0, TTX_i) < MAX_THW + 5 and min(TTX_0, TTX_i) > 0:
                        thw = (dis_to_x_0 - dis_to_x_i) / v_project_0

                        if thw > 0:
                            thw_list.append(thw)

    if len(thw_list) == 0:
        return None
    elif min(thw_list) > MAX_THW: 
        return None
    else:
        return min(thw_list)


def visualize(ttcs, vx_list, dx_list):
    vx_list = np.array(vx_list)
    plt.figure()
    ax1 = plt.subplot(121)
    plt.plot(ttcs, color="g", label="ttc")
    plt.plot(vx_list[:, 0], color="r", label="ego_vehicle")
    plt.plot(vx_list[:, 1], label="other_vehicle")
    plt.legend()
    ax1.set_title("velocity")
    ax2 = plt.subplot(122)
    plt.plot(dx_list)
    ax2.set_title("distance between two vehicles")
    plt.show()


# Output the video of the data with ttc less than ttc_threshold


def get_videos_of_small_ttc_cases(
    scenarioInstance, ttc_threshold, search_file_path, params_key_list, max_num=200
):
    scenarioInstance.set_save_video(True)
    scenarioInstance.set_save_data(False)

    import pandas as pd

    df = pd.read_csv(search_file_path, usecols=lambda x: x not in ["time"])
    df = df[df["ttc"] < ttc_threshold]
    # print the amout of data
    print("The amount of small ttc data is: ", len(df))
    # random select max_num rows
    df = df.sample(n=min(max_num, len(df)))
    for index, row in df.iterrows():
        params = [row[key] for key in params_key_list]
        scenarioInstance.run_step(*params)


# Use multiprocessing to perform PSO search


def multiprocessing_search_pso(
    scenarioInstance,
    weight_list,
    n_particles,
    max_iter,
    iw_list,
    c1_list,
    c2_list,
    seed_list,
    max_cpu_n=200,
    av_model_type="AV-I",
):
    print("Start multiprocessing parameter search")
    print("Total CPU count: ", cpu_count())

    n_cpus = min(cpu_count(), max_cpu_n)

    print("CPU allowed: ", n_cpus)

    name_prefix = scenarioInstance.name

    p = Pool(n_cpus)

    n_total = (
        len(weight_list) * len(iw_list) * len(c1_list) * len(c2_list) * len(seed_list)
    )
    print("Total tasks: ", n_total)

    # load av_model
    av_model = load_av_model(av_model_type)

    for seed in seed_list:
        for w in weight_list:
            for iw in iw_list:
                for c1 in c1_list:
                    for c2 in c2_list:
                        curr_name = (
                            "{}_w_{}_n_{}_m_{}_iw_{}_c1_{}_c2_{}_seed_{}".format(
                                name_prefix, w, n_particles, max_iter, iw, c1, c2, seed
                            )
                        )

                        # print("Running:{}".format(curr_name))

                        sc = copy.deepcopy(scenarioInstance)
                        sc.set_save_data(True)
                        sc.set_save_video(False)
                        sc.set_debug(False)
                        sc.set_name(curr_name)  
                        sc.set_random_seed(seed)  
                        sc.set_loss_weight([w / 10, 1 - w / 10, 50])  
                        sc.set_av_model(av_model) 
                        p.apply_async(
                            sc.search_pso, args=(n_particles, max_iter, iw, c1, c2)
                        )
    p.close()
    p.join()

    print("Finish multiprocessing parameter search\n")


# Read the results of the PSO parameter exploration experiment


def read_pso_search_files(output_path_base, file_name_prefix, output_natual_path):
    def get_parameters_from_file_name(filename):
        filename = filename.split("_")
        if "seed" in filename:
            return (
                float(filename[-13]),
                int(filename[-11]),
                int(filename[-9]),
                filename[-7],
                filename[-5],
                filename[-3],
                filename[-1],
            )
        else:
            return (
                float(filename[-11]),
                int(filename[-9]),
                int(filename[-7]),
                filename[-5],
                filename[-3],
                filename[-1],
            )

    # process natural data
    output_natual_path = Path(output_natual_path)
    try:
        natural_search_df = pd.read_csv(
            os.path.join(output_natual_path, "search_natural.csv")
        )
    except:
        natural_search_df = pd.DataFrame()
        print("search_natural.csv not found!")
    try:
        natural_statistic_df = pd.read_csv(
            os.path.join(output_natual_path, "statistic_natural.csv")
        )
    except:
        natural_statistic_df = pd.DataFrame()
        print("statistic_natural.csv not found!")

    output_path_list = os.listdir(output_path_base)
    # filter prefix
    output_path_list = list(
        filter(lambda x: x.startswith(file_name_prefix + "_"), output_path_list)
    )

    search_df = pd.DataFrame()
    statistic_df = pd.DataFrame()

    for output_path in output_path_list:
        output_path_system = os.path.join(output_path_base, output_path)
        for file in os.listdir(output_path_system):
            if file.endswith(".csv"):
                # get parameters from file name
                parameters = get_parameters_from_file_name(output_path)

                if parameters[0] not in [0, 3, 5, 7, 10]:
                    continue

                # use pandas to read csv
                df = pd.read_csv(os.path.join(output_path_system, file))
                # add parameters to dataframe
                df["w"] = round(parameters[0] / 10, 2)
                df["n"] = parameters[1]
                df["m"] = parameters[2]
                df["iw"] = parameters[3]
                df["c1"] = parameters[4]
                df["c2"] = parameters[5]
                df["seed"] = parameters[-1]
                # get search method, delete .csv
                search_method = file.split("_")[-1]
                search_method = search_method.split(".")[0]
                df["search_method"] = search_method
                # add to search_df if starts with search
                if file.startswith("search"):
                    # add index
                    df.reset_index(inplace=False, drop=False)
                    df["iter"] = df.index
                    search_df = pd.concat([search_df, df])
                # add to statistic_df if starts with statistic
                elif file.startswith("statistic"):
                    statistic_df = pd.concat([statistic_df, df])

    return search_df, statistic_df, natural_search_df, natural_statistic_df


def show_collision_rate(statistic_df, search_df, name, output_path, mode="normal"):

    if mode == "normal":

        # show collision rate group by w and seed
        collision_df = statistic_df.groupby(["w", "seed"])["collision"].sum()
        collision_df.reset_index()
        # distance_x_df from search_df
        distance_x_df = search_df.groupby(["w", "seed"])["distance_x"].sum() / 1000
        distance_x_df.reset_index
        # concat two df
        collision_rate_df = pd.concat([collision_df, distance_x_df], axis=1)
        collision_rate_df["collision_rate"] = (
            collision_rate_df["collision"] / collision_rate_df["distance_x"]
        )
        # sort by collision_rate
        collision_rate_df = collision_rate_df.sort_values(
            by=["collision_rate"], ascending=False
        )
        collision_rate_df = collision_rate_df.round(2)

        # collision_rate_df sort by w and seed
        collision_rate_df = collision_rate_df.sort_values(by=["w", "seed"])

        # collision_rate_mean_df: average collision_rate of each w
        collision_rate_mean_df = collision_rate_df.groupby(["w"])[
            "collision_rate"
        ].mean()
        # rename the second column name of collision_rate_mean_df
        collision_rate_mean_df = collision_rate_mean_df.rename("collision_rate_mean")
        collision_rate_mean_df = collision_rate_mean_df.reset_index()
        collision_rate_mean_df = collision_rate_mean_df.sort_values(by=["w"])

        # collision_rate_df: show collision rate group by w and seed
        # collision_rate_mean_df:show average collision rate of different weights
        print("Collision rate group by w and seed")
        print(collision_rate_mean_df)
        # save collision_rate_mean_df as csv file in output_path
        print("Save collision rate mean to {} directory".format(output_path))
        collision_rate_mean_df.to_csv(
            os.path.join(output_path, "collision_rate_mean.csv"), index=False
        )

        wandb_table = wandb.Table(
            columns=["w", "collision_rate_mean"],
            data=[
                [w, collision_rate_mean]
                for w, collision_rate_mean in zip(
                    collision_rate_mean_df["w"],
                    collision_rate_mean_df["collision_rate_mean"],
                )
            ],
        )
        # WanDBRunProvider.get_run().log(
        #     {"Table/Average collision rate (times/KM)": wandb_table}
        # )

        plt.rcParams["figure.figsize"] = [10, 5]
        plt.plot(
            collision_rate_mean_df["w"],
            collision_rate_mean_df["collision_rate_mean"],
            label="collision_rate_mean",
        )
        plt.xlabel("Weight")
        plt.ylabel("Collisions per KM")
        plt.title("Average collision rate in {}".format(name))
        print("Save collision rate line chart to {} directory".format(output_path))
        # WanDBRunProvider.get_run().log({"Image/Collision Rate": wandb.Image(plt)})
        plt.savefig(os.path.join(output_path, "collision_rate.png"))
        plt.close()

    elif mode == "hs":
        weight_list = search_df["w"].unique()
        weight_list = np.sort(weight_list)
        for weight in weight_list:
            # show collision rate in statistic_df, group by iw,c1,c2, result in collision_rate_df
            collision_df = (
                statistic_df[statistic_df["w"] == weight]
                .groupby(["iw", "c1", "c2"])["collision"]
                .sum()
            )
            collision_df.reset_index()
            # distance_x_df from search_df
            distance_x_df = (
                search_df[search_df["w"] == weight]
                .groupby(["iw", "c1", "c2"])["distance_x"]
                .sum()
                / 1000
            )
            distance_x_df.reset_index
            # concat two df
            collision_rate_df = pd.concat([collision_df, distance_x_df], axis=1)
            collision_rate_df["collision_rate"] = (
                collision_rate_df["collision"] / collision_rate_df["distance_x"]
            )
            # sort by collision_rate
            collision_rate_df = collision_rate_df.sort_values(
                by=["collision_rate"], ascending=False
            )
            collision_rate_df = collision_rate_df.round(2)

            # show results
            print("Collision rate group by w and seed")
            print(collision_rate_df)
            print("Save collision rate to {} directory".format(output_path))
            collision_rate_df.reset_index().to_csv(
                os.path.join(output_path, "collision_rate.csv"), index=False
            )


def show_follow_data(
    name,
    output_path,
    nat_df,
    statistic_df,
    search_df,
    criteria="ttc",
    max_value=100,
    min_value=0,
    max_count=900000,
):

    def filter_values(values, max_value=100, min_value=0, max_count=2000):
        new_values = [value for value in values if value is not None]
        new_values = [
            value for value in values if value <= max_value and value >= min_value
        ][:max_count]
        return new_values

    weight_list = search_df["w"].unique()
    weight_list = np.sort(weight_list)
    seed_list = search_df["seed"].unique()
    seed_list = np.sort(seed_list)

    col_len = len(seed_list)
    row_len = len(weight_list)
    plt.rcParams["figure.figsize"] = [3 * col_len, 3 * row_len]
    fig, ax = plt.subplots(row_len, col_len)

    for k, w in enumerate(weight_list):

        nat_data = filter_values(nat_df[criteria], max_value, min_value, max_count)
        # enumerate the seeds
        for j, seed in enumerate(seed_list):

            # opt_data: filter by w and seed
            opt_data = filter_values(
                statistic_df[(statistic_df["w"] == w) & (statistic_df["seed"] == seed)][
                    criteria
                ],
                max_value,
                min_value,
                max_count,
            )

            values, bins, _ = ax[k][j].hist(
                np.array(nat_data),
                bins=100,
                weights=np.ones_like(nat_data) / float(len(nat_data)),
                label="Nature sample",
                alpha=0.5,
                histtype="stepfilled",
                edgecolor="none",
            )
            area_nat = sum(np.diff(bins) * values)
            values, bins, _ = ax[k][j].hist(
                np.array(opt_data),
                bins=100,
                weights=np.ones_like(opt_data) / float(len(opt_data)),
                label="Optimized sample",
                alpha=0.5,
                histtype="stepfilled",
                edgecolor="none",
            )

        # set col and row header
        cols = ["seed={}".format(s) for s in seed_list]
        rows = ["w={}".format(w) for w in weight_list]
        for a, col in zip(ax[0], cols):
            a.set_title(col)
        for a, row in zip(ax[:, 0], rows):
            a.set_ylabel(row, rotation=90, size="large")

    # set for all
    handles, labels = ax[-1][-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle("{} distribution in {}".format(criteria.upper(), name), fontsize=15)
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    print("Save {} distribution to {} directory".format(criteria, output_path))
    # WanDBRunProvider.get_run().log({'Image/{} distribution in {}'.format(
    #     criteria.upper(), name): wandb.Image(plt)})
    plt.savefig(os.path.join(output_path, "{}.png".format(criteria)))
    plt.close()


def detect_convergence(loss_list, threshold=5, window_size=5):
    """Detects convergence in a sequence of loss values.
    detect_convergence takes an window_size argument that
    specifies the size of the sliding window to use for checking convergence.
    The function checks whether the difference between the current value and
    all values in the window is less than the threshold.
    """
    for i in range(window_size, len(loss_list)):
        if all(
            abs(loss_list[i] - x) <= threshold for x in loss_list[i - window_size : i]
        ):
            if loss_list[i] < 0:
                return i, loss_list[i]
    return -1, None


def show_loss_evaluation(name, search_df, output_path, mode="normal"):

    weight_list = search_df["w"].unique()
    weight_list = np.sort(weight_list)
    seed_list = search_df["seed"].unique()
    seed_list = np.sort(seed_list)

    if mode == "normal":
        criterias = ["loss"]
        col_len = len(seed_list)
        row_len = len(weight_list)
        plt.rcParams["figure.figsize"] = [3 * col_len, 3 * row_len]
        fig, ax = plt.subplots(row_len, col_len)

        for k, w in enumerate(weight_list):
            # enumerate the methods
            for i, seed in enumerate(seed_list):
                for criteria in criterias:
                    data = search_df[
                        (search_df["w"] == w) & (search_df["seed"] == seed)
                    ][criteria]

                    # draw the EMA wave of the criteria
                    data = data.ewm(span=10).mean()

                    ax[k][i].plot(data, label=criteria, alpha=0.6)

            # set col and row header
            cols = ["seed={}".format(s) for s in seed_list]
            rows = ["w={}".format(w) for w in weight_list]
            for a, col in zip(ax[0], cols):
                a.set_title(col)
            for a, row in zip(ax[:, 0], rows):
                a.set_ylabel(row, rotation=90, size="large")

        # set title for all
        handles, labels = ax[-1][-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")
        fig.suptitle("{} curve in {}".format(",".join(criterias), name), fontsize=15)
        fig.tight_layout(rect=[0, 0.03, 1, 0.98])

        print("Save Loss curve to {} directory".format(output_path))
        # WanDBRunProvider.get_run().log({
        #     'Image/{} curve in {}'.format(','.join(criterias), name): wandb.Image(plt)
        # })
        plt.savefig(os.path.join(output_path, "loss.png"))
        plt.close()

        return None

    elif mode == "hs":
        # create a new dataframe called result_df, cols are c1,c2,iw,iter,min_loss
        result_df = pd.DataFrame(columns=["iw", "c1", "c2", "iter", "min_loss"])
        for weight in weight_list:
            # plot loss curve in search_df, group by iw,c1,c2
            search_group_col = search_df[search_df["w"] == weight].groupby(["c1", "c2"])

            row_len = search_group_col.ngroups
            col_len = search_df[search_df["w"] == weight]["iw"].nunique()
            plt.rcParams["figure.figsize"] = [3 * col_len, 3 * row_len]
            fig, axs = plt.subplots(row_len, col_len)

            # plot in each group
            for i, ((c1, c2), group) in enumerate(search_group_col):
                group = group.sort_values(by=["iw"])
                for j, (value, group2) in enumerate(group.groupby(["iw"])):
                    # if value is a tuple, iw = value[0]
                    if isinstance(value, tuple):
                        iw = value[0]
                    else:
                        iw = value
                    group2 = group2.sort_values(by=["iter"])

                    # axs[i, j].plot(group2['iter'], group2['loss'])
                    axs[i, j].set_title("iw={}\nc1={},c2={}".format(iw, c1, c2))

                    # draw the EMA wave of the loss
                    group2["loss_ema"] = group2["loss"].ewm(span=10).mean()
                    axs[i, j].plot(group2["iter"], group2["loss_ema"])

                    # evaluate loss
                    iter, _ = detect_convergence(group2["loss"].values, 5)
                    min_loss = group2["loss"].min()
                    result_df = pd.concat(
                        [
                            result_df,
                            pd.Series(
                                {
                                    "c1": c1,
                                    "c2": c2,
                                    "iw": iw,
                                    "iter": iter,
                                    "min_loss": round(min_loss, 3),
                                }
                            )
                            .to_frame()
                            .T,
                        ],
                        ignore_index=True,
                    )

                fig.suptitle(
                    "{} curve in {}, w={}".format("Loss", name, weight), fontsize=15
                )
                fig.tight_layout(rect=[0, 0.03, 1, 0.98])

        # show result_df sort by min_loss and iter
        result_df = result_df.sort_values(by=["min_loss", "iter"])
        # reset index
        result_df = result_df.reset_index(drop=True)
        # show results
        print("loss evaluation of different hyperparameters")
        print(result_df)
        print("Save Loss evaluation to {} directory".format(output_path))
        result_df.to_csv(os.path.join(output_path, "loss.csv"), index=False)
        print("Save Loss curve to {} directory".format(output_path))
        # WanDBRunProvider.get_run().log({
        #     'Image/Loss curve in {}'.format(name): wandb.Image(plt)
        # })
        plt.savefig(os.path.join(output_path, "loss.png"))
        plt.close()

        # find the first row with min_loss and iter!=-1
        result_df = result_df[result_df["iter"] != -1]
        result_df = result_df.reset_index(drop=True)
        # if result_df is empty, return None
        if result_df.empty:
            print("Didn't find the converged hyperparameters")
            return None
        else:
            best_row = result_df.iloc[0]
            print("Best hyperparameters:\n{}".format(best_row))
            # WanDBRunProvider.get_run().log(
            #     {"Table/Convergence and minimum loss value of different hyperparameters": wandb.Table(dataframe=result_df)}
            # )
            return best_row


# Analyze the experimental results, output the collision rate line chart and the distribution chart of ttc and thw

# mode: normal or hs(hyperparameter search)


def analyze_results(
    output_path_base, file_name_prefix, output_natual_path, mode="normal"
):
    name = file_name_prefix
    output_path = output_path_base
    search_df, statistic_df, natural_search_df, natural_statistic_df = (
        read_pso_search_files(output_path_base, file_name_prefix, output_natual_path)
    )

    if mode == "normal":
        show_collision_rate(statistic_df, search_df, name, output_path, mode="normal")

        show_follow_data(
            name,
            output_path,
            natural_statistic_df,
            statistic_df,
            search_df,
            criteria="ttc",
            max_value=100,
            min_value=0,
            max_count=9999999,
        )
        show_follow_data(
            name,
            output_path,
            natural_statistic_df,
            statistic_df,
            search_df,
            criteria="thw",
            max_value=10,
            min_value=0,
            max_count=9999999,
        )
        show_loss_evaluation(name, search_df, output_path, mode)
        show_ttc_proportion(statistic_df, output_path)
    elif mode == "hs":
        # show_collision_rate(statistic_df, search_df, name,
        #                     output_path, mode="hs")
        return show_loss_evaluation(name, search_df, output_path, mode)


def load_av_model(av_model_type):
    # Load the AV model
    if av_model_type == "AV-I":
        av_model = None
    elif av_model_type == "AV-II-CROSS":
        # Fixed path
        av_model = PPO.load(Path("data/models/crossroads_av2_discrete_model"))
        print("AV-II-CROSS loaded!!")
    else:
        raise ValueError("Wrong AV model type!")
    return av_model


def show_ttc_proportion(statistic_df, output_path):
    # Group the dataframe by w and seed
    grouped_df = statistic_df.groupby(["w", "seed"])

    # Define a function to calculate the proportion of values within a range
    def proportion_within_range(x, lower, upper, filter_min, filter_max):
        total = ((x is not None) & (x >= filter_min) & (x <= filter_max)).sum()
        within_range = ((x >= lower) & (x <= upper)).sum()
        return round(within_range / total, 3)

    # Apply the function to calculate the proportion of values within the specified ranges for each group
    proportion_ttc = grouped_df["ttc"].apply(
        proportion_within_range, lower=0, upper=10, filter_min=0, filter_max=100
    )
    proportion_thw = grouped_df["thw"].apply(
        proportion_within_range, lower=0, upper=1, filter_min=0, filter_max=10
    )

    # Calculate the average of the proportions across different seeds for each value of w
    mean_proportion_ttc = proportion_ttc.groupby("w").mean()
    mean_proportion_thw = proportion_thw.groupby("w").mean()

    # Create a new dataframe to store the results
    result_df = pd.DataFrame(
        {
            "prop_ttc_[0,10](%)": mean_proportion_ttc * 100,
            "prop_thw_[0,1](%)": mean_proportion_thw * 100,
        }
    )

    # Sort the dataframe by w
    result_df = result_df.sort_values("w")

    # show results
    print("Show the proportion of intervals for TTC & THW")
    print(result_df)
    print("Save the proportion of intervals to {} directory".format(output_path))
    result_df.to_csv(os.path.join(output_path, "proportion.csv"), index=False)

    wandb_table = wandb.Table(
        columns=["w", "prop_ttc_[0,10](%)", "prop_thw_[0,1](%)"],
        data=[
            [w, prop_ttc, prop_thw]
            for w, prop_ttc, prop_thw in zip(
                result_df.index,
                result_df["prop_ttc_[0,10](%)"],
                result_df["prop_thw_[0,1](%)"],
            )
        ],
    )
    # WanDBRunProvider.get_run().log(
    #     {"Table/Percentage of intervals for TTC & THW": wandb_table}
    # )


def show_simulation_agents_data(agents_datas_df, output_path):
    # convert heading and steering to degree
    agents_datas_df["heading"] = agents_datas_df["heading"] * 180 / np.pi
    agents_datas_df["steering"] = agents_datas_df["steering"] * 180 / np.pi

    # replace id with 'EGO' if is_ego is True
    agents_datas_df.loc[agents_datas_df["is_ego"] == True, "id"] = "EGO"

    # save the agents_datas_df as csv file in output_path
    agents_datas_df.to_csv(
        os.path.join(output_path, "agents_data.csv"), index_label="index"
    )

    plt.rcParams["figure.figsize"] = [15, 10]
    # for each agent
    ids = agents_datas_df["id"].unique()
    for id in ids:
        df = agents_datas_df[agents_datas_df["id"] == id]
        df = df.reset_index(drop=True)
        df["index"] = df.index

        fig, ax = plt.subplots(2, 2)

        # vel
        ax[0][0].set_xlabel("step")
        ax[0][0].set_ylabel("m/s")
        ax[0][0].plot(df["vx"], label="vel_x", marker="o")
        ax[0][0].plot(df["vy"], label="vel_y", marker="o")
        ax[0][0].legend()

        # vel+acc
        ax2 = ax[0][1].twinx()
        ax[0][1].set_xlabel("step")
        # set the left y-axis label for ax[0][1]
        ax[0][1].set_ylabel("m/s")
        # set the right y-axis label for ax2
        ax2.set_ylabel("m/s^2")
        p1 = ax[0][1].plot(df["speed"], label="vel", color="orange", marker="o")
        p2 = ax2.plot(df["acceleration"], label="acc", color="magenta", marker="h")
        labs = [l.get_label() for l in p1 + p2]
        ax[0][1].legend(p1 + p2, labs)

        # location
        ax[1][0].set_xlabel("step")
        ax[1][0].set_ylabel("m")
        ax[1][0].plot(df["x"], label="x", marker="x")
        ax[1][0].plot(df["y"], label="y", marker="x")
        ax[1][0].legend()

        # heading+steering
        ax[1][1].set_xlabel("step")
        ax[1][1].set_ylabel("degree")
        ax[1][1].plot(df["heading"], label="heading", marker="s")
        ax[1][1].plot(df["steering"], label="steering", marker="s")
        ax[1][1].legend()

        fig.suptitle(f"Simulation Data of Veh {id}", fontsize=15)
        fig.tight_layout(rect=[0, 0.03, 1, 0.98])
        plt.savefig(os.path.join(output_path, "simulation_data_{}.png".format(id)))
        plt.close()

    # draw vel of all vehicles together
    fig, ax = plt.subplots(2, 1)
    plt.rcParams["figure.figsize"] = [15, 10]
    for id in ids:
        df = agents_datas_df[agents_datas_df["id"] == id]
        df = df.reset_index(drop=True)
        df["index"] = df.index
        ax[0].plot(df["speed"], label="veh_" + str(id), marker="o")
        ax[1].plot(df["acceleration"], label="veh_" + str(id), marker="h")
    leg = ax[0].legend()
    leg = ax[1].legend()
    ax[0].set_xlabel("step")
    ax[0].set_ylabel("m/s")
    ax[0].set_title("Velocity")

    ax[1].set_xlabel("step")
    ax[1].set_ylabel("m/s^2")
    ax[1].set_title("Acceleration")

    fig.suptitle("Simulation Data of All Vehicles", fontsize=15)
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])

    plt.savefig(os.path.join(output_path, "simulation_data_all.png".format(id)))
    plt.close()


def change_route_value(file_path, route_id, param_name, value):
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Traverse all 'route' elements
    for route in root.findall(".//route"):
        # Check if the 'id' attribute of the element is the given route_id
        if route.get("id") == str(route_id):
            # Traverse all 'scenarios' elements under the 'route' element with 'id' as the given route_id
            for scenarios in route.findall("scenarios"):
                # Traverse all 'scenario' elements under the 'scenarios' element
                for scenario in scenarios.findall("scenario"):
                    # Traverse all param_name elements under the 'scenario' element
                    for param in scenario.findall(param_name):
                        # Check if the 'value' attribute of the element exists
                        if param.get("value"):
                            # If so, modify the value of the 'value' attribute to the given value
                            param.set("value", str(value))

    # Save the modified XML data back to the file
    tree.write(file_path)


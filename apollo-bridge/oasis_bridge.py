#!/usr/bin/env python
#
# SYNKROTRON Confidential
# Copyright (C) 2023 SYNKROTRON Inc. All rights reserved.
# The source code for this program is not published
# and protected by copyright controlled
#


import os
import ast
import json
import time
import etcd
import yaml
import pathlib
import multiprocessing

from utils.logurus import init_log
from utils.redis_util import RedisUtil
from utils.global_args import GlobalArgs
from utils.response2etcd import ResponseToEtcd

from process_task import ProcessTask


root_dir = pathlib.Path(__file__).resolve().parent


class OasisBridge:
    def __init__(self, log):
        self.log = log
        GlobalArgs(self.log)

        config_file = os.path.dirname(os.path.abspath(__file__)) + "/config/settings.yaml"
        with open(config_file, encoding="utf-8") as f:
            self.parameters = yaml.safe_load(f)["oasis"]

        # redis and etcd init
        # self.rs = RedisUtil(self.parameters["host"], self.parameters["redis_port"],
        #                     self.parameters["redis_password"], self.parameters["redis_db"])
        # self.etcd_client = etcd.Client(host=self.parameters["host"], port=int(self.parameters["etcd_port"]))

        # drive_mode = self.parameters["drive_mode"]
        # drive_version = self.parameters["drive_version"]
        # self.tasks_key = f"/oasis/{drive_mode}/{drive_version}/{self.parameters['host']}/tasks/"

        # module = self.parameters['module']
        # self.response = ResponseToEtcd(module, self.parameters['host'], self.etcd_client)

        # self.log.info(f"==================== oasis bridge init ====================")
        # self.log.info(f"tasks_key is {self.tasks_key}")
        # self.log.debug(f"args is \n{json.dumps(self.parameters, indent=4)}")

    def run(self):

        # 通过redis通信
        # channel = self.tasks_key.rstrip("/")
        # pubsub = self.rs.pubsub()
        # self.rs.subscribe(pubsub, channel)
        # self.log.info(f"channel is {channel}")

        # task_status_dict = multiprocessing.Manager().dict() # 就单纯状态字典

        # 根据受到的消息，启动任务
        # for message in self.rs.listen(pubsub):
        #     if message['type'] == 'message':
        #         content_dict = ast.literal_eval(message['data'])
        #         self.log.info(f"content_dict is {content_dict}")
        #         task_id = str(content_dict.get("task_id"))
        #         status = content_dict.get("status")
        #         carla_info = content_dict.get("carla_info")

        #         if status == "init":  # 下发初始化动作
        #             # self.log.info(f"receive task manager action is {status}")
        #             # task_status_dict[task_id] = "init"
        #             self.start_process(task_id, carla_info, task_status_dict)
        #             # self.log.info(f"task_status_dict is {task_status_dict}")

                # elif status in ["stop", "delete"]:
                #     # task_status_dict[task_id] = "stop"
                #     self.response.log(task_id, "vehicle_control_system_stopped", "status")
                #     self.log.info(f"receive task manager action is {status}")
                #     # self.log.info(f"task_status_dict is {task_status_dict}")
                #     # 确保 process task 进程收到 stop 消息
                #     time.sleep(0.5)
                #     # task_status_dict.clear()
                # else:
                #     self.log.error("task manager send status not match apollo bridge")


        # 直接启动！
        self.start_process(task_id, carla_info, task_status_dict)

    def start_process(self, task_id, carla_info, task_status_dict):
        task_args = (task_id, carla_info, self.rs, self.log, self.parameters, self.response, task_status_dict)
        task_process = multiprocessing.Process(target=run_task, args=task_args)
        task_process.start()


def run_task(task_id, carla_info, rs, log, parameters, response, task_status_dict):
    process_task = ProcessTask(task_id, carla_info, rs, log, parameters, task_status_dict)
    log.info("report to etcd ads is running")
    nohup_file_path = os.path.join(root_dir, "nohup.out")
    if os.path.exists(nohup_file_path):
        os.remove(nohup_file_path)
    response.log(task_id, "vehicle_control_system_running", "status")
    process_task.run_apollo()


if __name__ == "__main__":
    log_dir = str(root_dir.joinpath("log"))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, "oasis_bridge.log")
    logger = init_log(log_file)
    OasisBridge(logger).run()

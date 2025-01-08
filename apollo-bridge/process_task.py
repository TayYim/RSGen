#!/usr/bin/env python
"""
/*******************************************************************************
 * SYNKROTRON Confidential
 * Copyright (C) 2023 SYNKROTRON Inc. All rights reserved.
 * The source code for this program is not published
 * and protected by copyright controlled
 *******************************************************************************/
"""

import os
import time
import json
import carla

import subprocess

from sensor.imu import IMU
from sensor.gnss import GNSS
from sensor.lidar import Lidar
from sensor.camera import Camera
from sensor.obstacle import Obstacle

from utils.map_util import MapUtil
from utils.cyber_node import CyberNode

from core.routing import Routing
from actor.ego_vehicle import EgoVehicle

from cyber.proto.clock_pb2 import Clock
from modules.common_msgs.routing_msgs import routing_pb2


class ProcessTask(CyberNode):
    def __init__(self, carla_info, log, parameters):
        super().__init__("oasis_bridge")
        self.log = log
        self.args = parameters

        # Get map name and destination position
        self.map_name = self.task_info["scenario_param"]["map_name"]
        self.dst_position = self.task_info["scenario_param"]["ego_start_and_end_position"]["end_position"]

        self.map_util = None
        self.ego_vehicle = None
        # True Indicates that all signal lights are running properly. Otherwise, all signal lights remain Off
        self.signal_lights_on = False

        self.clock_writer = self.create_writer("/clock", Clock, 10)
        self.routing_req = self.create_writer("/apollo/routing_request", routing_pb2.RoutingRequest)
        self.routing_res = self.create_writer("/apollo/routing_response", routing_pb2.RoutingResponse)

        # carla info
        self.carla_host = carla_info["ip"]
        self.carla_port = int(carla_info["port"])
        self.carla_timeout = self.args["carla_timeout"]
        self.client = carla.Client(host=self.carla_host, port=self.carla_port)
        self.client.set_timeout(self.carla_timeout)
        self.world = self.client.get_world()


    def run_apollo(self):
        self.log.info("run apollo")

        # 每次运行任务时开启 control 模块，运行完毕后关闭 control 模块
        self.start_apollo_control(self.log)

        self._set_map(f"carla_{self.map_name.lower()}")

        # Wait for Carla until the ego vehicle spawned
        self._find_ego_vehicle()

        # Send routing response to run the task
        self._set_destination(self.dst_position, use_carla_routing=False)

        # self._set_sensors()
        self._set_signal_lights()

        self._update_actor_factor()

    def _set_map(self, map_name):
        """Set map in apollo"""
        self.map_util = MapUtil(map_name)
        file_name = "/apollo/modules/common/data/global_flagfile.txt"
        new_line = f"--map_dir=/apollo/modules/map/data/{map_name}"
        with open(file_name, "r", encoding="utf-8") as file:
            lines = file.readlines()

        lines[-1] = new_line + "\n"

        with open(file_name, "w", encoding="utf-8") as file:
            file.writelines(lines)

    def _set_destination(self, dst_position, use_carla_routing=False):
        """
        Set destination and send routing request or response.
        If you use_carla_routing is True, GlobalRoutePlanner will be used to plan the global route
        and send RoutingResponse to Apollo.
        Else RoutingRequest will be sent to Apollo.
        Args:
            dst_position: The destination point extract from task_info
            use_carla_routing: Whether to use carla routing function. (GlobalRoutePlanner)
        """
        routing = Routing(self.world.get_map(), self.map_util)
        start_location = self.ego_vehicle.get_location()
        target_location = carla.Location(
            float(dst_position["x"]),
            float(dst_position["y"]) * -1,
            float(dst_position["z"]),
        )
        if use_carla_routing:
            routing_response = routing.generate_routing_response(start_location, target_location)
            self.routing_res.write(routing_response)
        else:
            routing_request = routing.generate_routing_request(start_location, target_location)
            self.routing_req.write(routing_request)

    def _find_ego_vehicle(self, timeout=60):
        """
        Wait for the ego vehicle to be generated, and then obtain the ego vehicle.
        """
        start_time = time.time()
        while self.ego_vehicle is None and time.time() - start_time < timeout:
            actor_list = self.world.get_actors()
            for actor in actor_list:
                if actor.attributes.get("role_name") == "ego_vehicle":
                    self.ego_vehicle = actor
                    break

        if self.ego_vehicle is None:
            raise TimeoutError(f"Finding ego vehicle timeout in {timeout} seconds.")
        else:
            self.log.info(f"ego_vehicle id: {self.ego_vehicle.id}")

    def _find_gnss_actor(self):
        while True:
            actors = self.world.get_actors().filter("sensor.other.gnss")
            gnss_role_name = self.args.get("ego_sensors").get("gnss").get("role_name")
            for actor in actors:
                if gnss_role_name == actor.attributes.get('role_name'):
                    self.log.info("gnss sensor found")
                    return actor
                self.log.warning("not found gnss sensor")
            time.sleep(0.1)

    def _find_imu_actor(self):
        while True:
            actors = self.world.get_actors().filter("sensor.other.imu")
            gnss_role_name = self.args.get("ego_sensors").get("imu").get("role_name")
            for actor in actors:
                if gnss_role_name == actor.attributes.get('role_name'):
                    self.log.info("imu sensor found")
                    return actor
                self.log.warning("not found imu sensor")
            time.sleep(0.1)

    def _find_lidar_actor(self):
        while True:
            actors = self.world.get_actors().filter("sensor.lidar.ray_cast")
            gnss_role_name = self.args.get("ego_sensors").get("lidar").get("role_name")
            for actor in actors:
                if gnss_role_name == actor.attributes.get('role_name'):
                    self.log.info("lidar sensor found")
                    return actor
                self.log.warning("not found lidar sensor")
            time.sleep(0.1)

    def _find_front_6mm_camera(self):
        while True:
            actors = self.world.get_actors().filter("sensor.camera.rgb")
            gnss_role_name = self.args.get("ego_sensors").get("front_6mm_camera").get("role_name")
            for actor in actors:
                if gnss_role_name == actor.attributes.get('role_name'):
                    self.log.info("front_6mm_camera sensor found")
                    return actor
                self.log.warning("not found front_6mm_camera sensor")
            time.sleep(0.1)

    def _find_front_12mm_camera(self):
        while True:
            actors = self.world.get_actors().filter("sensor.camera.rgb")
            gnss_role_name = self.args.get("ego_sensors").get("front_12mm_camera").get("role_name")
            for actor in actors:
                if gnss_role_name == actor.attributes.get('role_name'):
                    self.log.info("front_12mm_camera sensor found")
                    return actor
                self.log.warning("not found front_12mm_camera sensor")
            time.sleep(0.1)

    def _set_signal_lights(self):
        if not self.signal_lights_on:
            # turn off all signals
            traffic_lights = self.world.get_actors().filter("traffic.traffic_light")
            for traffic_light in traffic_lights:
                traffic_light.set_state(carla.TrafficLightState.Off)
                traffic_light.freeze(True)

    def _update_actor_factor(self):
        # 先找 Carla 对象
        gnss_actor = self._find_gnss_actor()
        imu_actor = self._find_imu_actor()
        lidar_actor = self._find_lidar_actor()
        front_6mm_actor = self._find_front_6mm_camera()

        GNSS(gnss_actor, self.ego_vehicle, self)
        IMU(imu_actor, self.ego_vehicle, self)
        Lidar(lidar_actor, self.ego_vehicle, self)
        # 用一个摄像头测试感知
        Camera(front_6mm_actor, "front_6mm", self)
        ego_vehicle_obj = EgoVehicle(self.ego_vehicle, self)

        # 启动雷达感知时，首次需要给 /apollo/perception/obstacles channel 写入空数据，否则 planning 模块报错
        obstacle_obj = Obstacle("obstacle", self.world, self)
        obstacle_obj.update_null_obstacle()

        perception_switch = self.args.get("perception_switch")
        self.log.warning(f"perception_switch is {perception_switch}, bool type is {bool(perception_switch)}")

        while True:
            frame = self.world.SendTickMsg()
            self.clock_writer.write(Clock(clock=int(self.get_time())))

            ego_vehicle_obj.update()

            # 如果感知开关关闭，即仅测试规控，则需要将真实的障碍物更新到 /apollo/perception/obstacles 中
            if perception_switch is False:
                obstacle_obj.update_truth_obstacle()

            # 场景结束，退出进程
            task_status = self.task_status_dict.get(self.task_id)
            if task_status == "stop":
                self.task_status_dict.pop(self.task_id)
                break

        # 每次运行任务时开启 control 模块，运行完毕后关闭 control 模块
        self.stop_apollo_control(self.log)
        self.log.info(f"receive task manager action is {task_status}, task process exit")




#!/usr/bin/env python

import os
import time
import json
import carla
import sys

import subprocess
import psutil

from sensor.imu import IMU
from sensor.gnss import GNSS
from sensor.lidar import Lidar
from sensor.camera import Camera
from sensor.obstacle import Obstacle

from utils.map_util import MapUtil
from utils.cyber_node import CyberNode

from core.routing import Routing
from actor.ego_vehicle import EgoVehicle

from utils.logurus import init_log

from cyber.proto.clock_pb2 import Clock
from modules.common_msgs.routing_msgs import routing_pb2
import yaml
import signal
from dreamview_carla.dreamview import Connection
import time
import argparse


class OSGBridge(CyberNode):

    def __init__(self, params, args, carla_world, log):
        super().__init__("OSG_Carla_Cyber_Bridge")

        self.params = params
        self.log = log
        self.args = args

        # Get map name and destination position
        if args.map:
            self.map_name = args.map
        else:
            self.map_name = params["town"]

        if args.dest_x:
            self.dest_trans = carla.Transform(carla.Location(x=args.dest_x, y=args.dest_y, z=args.dest_z), carla.Rotation(yaw=args.dest_yaw, pitch=0, roll=0))
        else:
            # self.dest_trans = carla.Transform(carla.Location(x=396.3, y=90, z=0), carla.Rotation(yaw=-90, pitch=0, roll=0))
            # self.dest_trans = carla.Transform(carla.Location(x=315.3, y=199.16, z=0), carla.Rotation(yaw=0, pitch=0, roll=0))
            
            # Town04
            # original
            # self.dest_trans = carla.Transform(carla.Location(x=-9.15, y=-9, z=0), carla.Rotation(yaw=90, pitch=0, roll=0))
            # On bridge
            # self.dest_trans = carla.Transform(carla.Location(x=30, y=13.23, z=12), carla.Rotation(yaw=-180, pitch=0, roll=0))
            self.dest_trans = carla.Transform(carla.Location(x=-215.35, y=12.66, z=4.3), carla.Rotation(yaw=-180, pitch=0, roll=0))


        self.map_util = None
        self.ego_vehicle = None
        # True Indicates that all signal lights are running properly. Otherwise, all signal lights remain Off
        self.signal_lights_on = False

        self.clock_writer = self.create_writer("/clock", Clock, 10)
        self.routing_req = self.create_writer("/apollo/routing_request", routing_pb2.RoutingRequest)
        self.routing_res = self.create_writer("/apollo/routing_response", routing_pb2.RoutingResponse)

        self.world = carla_world

        self.dreamview_connection = None

        self.control_start_flag = False

        self.control_queue = [] # To store control command


    def run_apollo(self):
        # Wait for Carla until the ego vehicle spawned
        self._find_ego_vehicle() # TODO: role name should come from params
        self.log.info("Ego found")

        self.world.tick()

        # connect to dreamview
        self.dreamview_connection = Connection(self.ego_vehicle, port="8899", log=self.log)

        # set map
        self.dreamview_connection.set_hd_map(f"carla_{self.map_name.lower()}")
    

        # Set dest
        modules = ['Prediction', 'Control', 'Planning']
        # self.dreamview_connection.set_destination_tranform(self.dest_trans)
        self.dreamview_connection.enable_apollo(self.dest_trans, modules)


        # self._set_sensors()
        self._set_signal_lights()
        self.log.info("Signal set")

        self.log.info("Start to update actor factor")
        self._update_actor_factor()

    def _find_ego_vehicle(self, timeout=60):
        """
        Wait for the ego vehicle to be generated, and then obtain the ego vehicle.
        """
        start_time = time.time()
        while self.ego_vehicle is None and time.time() - start_time < timeout:
            actor_list = self.world.get_actors()
            for actor in actor_list:
                if actor.attributes.get("role_name") in self.params.get('ego_vehicle').get('role_name'):
                    self.ego_vehicle = actor
                    break

        if self.ego_vehicle is None:
            raise TimeoutError(f"Finding ego vehicle timeout in {timeout} seconds.")
        else:
            self.log.info(f"ego_vehicle id: {self.ego_vehicle.id}")

    def _find_gnss_actor(self):
        while True:
            actors = self.world.get_actors().filter("sensor.other.gnss")
            gnss_role_name = self.params.get("ego_sensors").get("gnss").get("role_name")
            for actor in actors:
                if gnss_role_name == actor.attributes.get('role_name'):
                    self.log.info("gnss sensor found")
                    return actor
                self.log.warning("not found gnss sensor")
            time.sleep(0.1)

    def _find_imu_actor(self):
        while True:
            actors = self.world.get_actors().filter("sensor.other.imu")
            gnss_role_name = self.params.get("ego_sensors").get("imu").get("role_name")
            for actor in actors:
                if gnss_role_name == actor.attributes.get('role_name'):
                    self.log.info("imu sensor found")
                    return actor
                self.log.warning("not found imu sensor")
            time.sleep(0.1)

    def _find_lidar_actor(self):
        while True:
            actors = self.world.get_actors().filter("sensor.lidar.ray_cast")
            gnss_role_name = self.params.get("ego_sensors").get("lidar").get("role_name")
            for actor in actors:
                if gnss_role_name == actor.attributes.get('role_name'):
                    self.log.info("lidar sensor found")
                    return actor
                self.log.warning("not found lidar sensor")
            time.sleep(0.1)

    def _find_front_6mm_camera(self):
        while True:
            actors = self.world.get_actors().filter("sensor.camera.rgb")
            gnss_role_name = self.params.get("ego_sensors").get("front_6mm_camera").get("role_name")
            for actor in actors:
                if gnss_role_name == actor.attributes.get('role_name'):
                    self.log.info("front_6mm_camera sensor found")
                    return actor
                self.log.warning("not found front_6mm_camera sensor")
            time.sleep(0.1)

    def _set_signal_lights(self):
        if not self.signal_lights_on:
            # turn off all signals
            traffic_lights = self.world.get_actors().filter("traffic.traffic_light")
            for traffic_light in traffic_lights:
                traffic_light.set_state(carla.TrafficLightState.Off)
                traffic_light.freeze(True)

    def _update_actor_factor(self):
        # First find the Carla object
        gnss_actor = self._find_gnss_actor()
        imu_actor = self._find_imu_actor()
        lidar_actor = self._find_lidar_actor()
        front_6mm_actor = self._find_front_6mm_camera()

        GNSS(gnss_actor, self.ego_vehicle, self)
        IMU(imu_actor, self.ego_vehicle, self)
        Lidar(lidar_actor, self.ego_vehicle, self)
        # Use a camera to test perception
        Camera(front_6mm_actor, "front_6mm", self)
        ego_vehicle_obj = EgoVehicle(self.ego_vehicle, self)

        # When starting the radar perception, you need to write empty data to the /apollo/perception/obstacles channel for the first time, otherwise the planning module will report an error
        obstacle_obj = Obstacle("obstacle", self.world, self)
        # obstacle_obj.update_null_obstacle()

        perception_switch = self.params.get("perception_switch")
        self.log.info(f"Perception Switch: {perception_switch}")

        def on_exit(sig, frame):
            # self.stop_apollo_control(self.log)
            self.log.info("Exit!")
            self.dreamview_connection.disable_module("Control")
            self.dreamview_connection.disable_module("Prediction")
            self.dreamview_connection.disable_module("Planning")
            self.dreamview_connection.disconnect()
            sys.exit(0)

        signal.signal(signal.SIGINT, on_exit)
        signal.signal(signal.SIGTERM, on_exit)

        self.world.tick()

        while self.ego_vehicle.is_alive:

            frame = self.world.tick()

            world_snapshot = self.world.get_snapshot()
            self.timestamp = self.get_timestamp(
                world_snapshot.timestamp.elapsed_seconds, from_sec=True
            )
            self.clock_writer.write(
                Clock(clock=int(self.timestamp["secs"]))
            )


            # update control command
            # if len(self.control_queue)==0:
            #     pass
            # else:
            #     curr_control = self.control_queue[0]
            #     self.control_queue.clear()
            #     ego_vehicle_obj.apply_control(curr_control)


            ego_vehicle_obj.update()

            # If the perception switch is turned off, that is, only the rule control is tested, the real obstacles need to be updated to /apollo/perception/obstacles
            if perception_switch is False:
                obstacle_obj.update_truth_obstacle()
            
def is_process_running(process_name):
    """
    Check if the process with the specified name is running
    """
    # Enumerate all running processes
    for proc in psutil.process_iter(attrs=['name']):
        try:
            # Find the process name that matches the specified name
            if process_name.lower() in proc.info['name'].lower():
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return False

def start_dreamview():
    """
    Start Dreamview by executing the script
    """
    # First clear all mainboard processes
    subprocess.run(['pkill', '-9', 'mainboard'])
    # Run the bootstrap script to start Dreamview
    subprocess.run(['/apollo/scripts/bootstrap.sh', 'restart'], shell=True)



def main(args):
    """
    main function for carla simulator Cyber bridge
    maintaining the communication client and the CarlaBridge object
    """
    log = init_log("bridge.log")
    carla_client = None
    

    config_file = os.path.dirname(os.path.abspath(__file__)) + "/config/settings.yaml"
    with open(config_file, encoding="utf-8") as f:
        parameters = yaml.safe_load(f)
    params = parameters["carla"]

    log.info(
        f"Trying to connect to {params['host']}:{params['port']}"
    )
    

    try:
        # Connect to Carla
        carla_client = carla.Client(
            host=params["host"], port=params["port"]
        )
        carla_client.set_timeout(params["timeout"])

        carla_world = carla_client.get_world()

        log.info(f"Connect to {params['host']}:{params['port']} successfully.")

        carla_world.tick()

        # Check dreamview status
        # If dreamview is not running, start it
        if not is_process_running('dreamview'):
            log.info("Dreamview not started. Starting Dreamview...")
            start_dreamview()
        else:
            log.info("Dreamview is already running.")

        carla_bridge = OSGBridge(params, args, carla_world, log)

        carla_bridge.run_apollo()

    except (IOError, RuntimeError) as e:
        log.error(f"Error: {e}")
    except KeyboardInterrupt as e:
        log.error(f"Error: {e}")
    except Exception as e:  # pylint: disable=W0718
        log.error(e)
    finally:
        # carla_world = carla_client.get_world()
        # settings = carla_world.get_settings()
        # settings.synchronous_mode = False
        # settings.fixed_delta_seconds = None
        # carla_world.apply_settings(settings)
        log.warning("Shutting down.")
        # carla_bridge.destroy()
        del carla_world
        del carla_client


if __name__ == "__main__":
    # Create a parser
    parser = argparse.ArgumentParser(description="Run OSG bridge.")
    
    parser.add_argument('-m', '--map', type=str, help='Town name')
    
    parser.add_argument('-x', '--dest_x', type=float, help='x')
    parser.add_argument('-y', '--dest_y', type=float, help='y')
    parser.add_argument('-z', '--dest_z', type=float, help='z')
    parser.add_argument('-yaw', '--dest_yaw', type=float, help='yaw')
    
    args = parser.parse_args()
    
    main(args)


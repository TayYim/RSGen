#!/usr/bin/env python

import utils.transforms
from utils.global_args import GlobalArgs

from modules.common_msgs.sensor_msgs.ins_pb2 import InsStat
from modules.common_msgs.localization_msgs.gps_pb2 import Gps
from modules.common_msgs.sensor_msgs.heading_pb2 import Heading
from modules.common_msgs.sensor_msgs.gnss_best_pose_pb2 import GnssBestPose


class GNSS:
    def __init__(self, actor, ego_vehicle, node):
        self.actor = actor
        self.node = node
        self.log = GlobalArgs.log
        self.ego_vehicle = ego_vehicle

        self.gnss_best_pose_writer = self.node.create_writer("/apollo/sensor/gnss/best_pose", GnssBestPose, qos_depth=10)
        self.gnss_odometry_writer = self.node.create_writer("/apollo/sensor/gnss/odometry", Gps, qos_depth=10)
        self.gnss_heading_writer = self.node.create_writer("/apollo/sensor/gnss/heading", Heading, qos_depth=10)
        self.gnss_status_writer = self.node.create_writer("/apollo/sensor/gnss/ins_stat", InsStat, qos_depth=10)

        self.actor.listen(self._sensor_data_updated)

    def _sensor_data_updated(self, carla_gnss_measurement):
        now_cyber_time = self.node.get_time()
        frame_id = "ego_vehicle/gnss"

        gnss_best_pose_msg = GnssBestPose()
        gnss_best_pose_msg.header.timestamp_sec = now_cyber_time
        gnss_best_pose_msg.header.module_name = "gnss"
        gnss_best_pose_msg.header.frame_id = frame_id
        gnss_best_pose_msg.latitude = carla_gnss_measurement.latitude
        gnss_best_pose_msg.longitude = carla_gnss_measurement.longitude
        gnss_best_pose_msg.height_msl = carla_gnss_measurement.altitude
        self.gnss_best_pose_writer.write(gnss_best_pose_msg)

        gnss_odometry_msg = Gps()
        gnss_odometry_msg.header.timestamp_sec = now_cyber_time
        gnss_odometry_msg.header.module_name = "gnss"
        gnss_odometry_msg.header.frame_id = frame_id

        cyber_pose = utils.transforms.carla_transform_to_cyber_pose(self.ego_vehicle.get_transform())
        gnss_odometry_msg.localization.CopyFrom(cyber_pose)
        self.gnss_odometry_writer.write(gnss_odometry_msg)

        gnss_heading_msg = Heading()
        gnss_heading_msg.header.timestamp_sec = now_cyber_time
        gnss_heading_msg.header.module_name = "gnss"
        gnss_heading_msg.header.frame_id = frame_id
        gnss_heading_msg.measurement_time = now_cyber_time
        _, _, yaw = utils.transforms.carla_rotation_to_rpy(self.actor.get_transform().rotation)
        gnss_heading_msg.heading = yaw
        self.gnss_heading_writer.write(gnss_heading_msg)

        gnss_status_msg = InsStat()
        gnss_status_msg.header.timestamp_sec = now_cyber_time
        gnss_status_msg.header.module_name = "gnss"
        gnss_status_msg.ins_status = 0
        gnss_status_msg.pos_type = 56
        self.gnss_status_writer.write(gnss_status_msg)

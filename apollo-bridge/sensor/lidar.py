#!/usr/bin/env python

import time
import numpy as np

from utils.global_args import GlobalArgs
from modules.common_msgs.basic_msgs.header_pb2 import Header
from modules.common_msgs.sensor_msgs.pointcloud_pb2 import PointXYZIT, PointCloud


class Lidar:
    def __init__(self,  actor, ego_vehicle, node):
        self.node = node
        self.actor = actor
        self.log = GlobalArgs.log
        self.ego_vehicle = ego_vehicle

        self.lidar_writer = self.node.create_writer("/apollo/sensor/velodyne128/compensator/PointCloud2",
                                                    PointCloud,
                                                    qos_depth=10)

        # self.lidar_writer = self.node.create_writer("/apollo/sensor/lidar128/compensator/PointCloud2",
        #                                             PointCloud,
        #                                             qos_depth=10)

        self.actor.listen(self._sensor_data_updated)

    def _sensor_data_updated(self, carla_lidar_measurement):
        lidar_data = np.fromstring(bytes(carla_lidar_measurement.raw_data), dtype=np.float32)
        lidar_data = np.reshape(lidar_data, (int(lidar_data.shape[0] / 4), 4))
        # we take the opposite of y-axis
        # (as lidar point are express in left-handed coordinate system, and ros need right-handed)
        lidar_data[:, 1] *= -1

        # lidar_data = lidar_data[:, :3]
        # lidar_data[:, 0:2] *= -1
        # we also need to permute x and y
        # lidar_data = lidar_data[..., [1, 0, 2]]
        header = self.get_msg_header()
        point_cloud_msg = PointCloud()
        point_cloud_msg.header.CopyFrom(header)
        point_cloud_msg.header.frame_id = self.actor.attributes['role_name']
        point_cloud_msg.frame_id = self.actor.attributes['role_name']
        point_cloud_msg.measurement_time = self.node.get_time()

        for lidar_point in lidar_data:
            cyber_point = PointXYZIT()
            cyber_point.x = lidar_point[0]
            cyber_point.y = lidar_point[1]
            cyber_point.z = lidar_point[2]
            point_cloud_msg.point.append(cyber_point)

        self.lidar_writer.write(point_cloud_msg)

    def get_msg_header(self, frame_id=None, timestamp=None):
        header = Header()
        if frame_id:
            header.frame_id = frame_id
        else:
            header.frame_id = "lidar"

        if not timestamp:
            timestamp = self.node.get_time()
        header.timestamp_sec = timestamp
        return header

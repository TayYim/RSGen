#!usr/bin/env python

import math

from utils.global_args import GlobalArgs
from modules.common_msgs.localization_msgs.imu_pb2 import CorrectedImu


class IMU:
    def __init__(self, actor, ego_vehicle, node):
        self.actor = actor
        self.node = node
        self.log = GlobalArgs.log
        self.ego_vehicle = ego_vehicle

        self.imu_writer = self.node.create_writer("/apollo/sensor/gnss/imu", CorrectedImu, qos_depth=20)

        self.actor.listen(self._sensor_data_updated)

    def _sensor_data_updated(self, carla_imu_measurement):
        imu_msg = CorrectedImu()
        imu_msg.header.timestamp_sec = self.node.get_time()
        imu_msg.header.frame_id = "ego_vehicle/default"

        # Carla uses a left-handed coordinate convention (X forward, Y right, Z up).
        # Here, these measurements are converted to the right-handed ROS convention
        #  (X forward, Y left, Z up).
        imu_msg.imu.angular_velocity.x = -carla_imu_measurement.gyroscope.x
        imu_msg.imu.angular_velocity.y = carla_imu_measurement.gyroscope.y
        imu_msg.imu.angular_velocity.z = -carla_imu_measurement.gyroscope.z

        imu_msg.imu.linear_acceleration.x = carla_imu_measurement.accelerometer.x
        imu_msg.imu.linear_acceleration.y = -carla_imu_measurement.accelerometer.y
        imu_msg.imu.linear_acceleration.z = carla_imu_measurement.accelerometer.z

        imu_msg.imu.euler_angles.x = (carla_imu_measurement.transform.rotation.roll / 180 * math.pi)
        imu_msg.imu.euler_angles.y = (carla_imu_measurement.transform.rotation.pitch / 180 * math.pi)
        imu_msg.imu.euler_angles.z = (carla_imu_measurement.transform.rotation.yaw / 180 * math.pi)

        self.imu_writer.write(imu_msg)

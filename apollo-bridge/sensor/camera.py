#!/usr/bin/env python

#
# Copyright (c) 2018-2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
#

import cv2
import math
import numpy as np

from utils.global_args import GlobalArgs
from modules.common_msgs.basic_msgs.header_pb2 import Header
from modules.common_msgs.sensor_msgs.sensor_image_pb2 import CompressedImage


class CameraInfo:
    def __init__(self):
        self.height = 0
        self.width = 0
        self.distortion_model = ''
        self.D = []
        self.K = []
        self.R = []
        self.P = []
        self.binning_x = 0
        self.binning_y = 0


class Camera:
    """
    Sensor implementation details for cameras
    """
    def __init__(self, actor, name, node):
        self.node = node
        self.actor = actor
        self.log = GlobalArgs.log
        self.name = name

        self._build_camera_info()
        self.compressed_6mm_writer = node.create_writer("/apollo/sensor/camera/front_6mm/image/compressed",
                                                        CompressedImage,
                                                        qos_depth=10)
        self.compressed_12mm_writer = node.create_writer("/apollo/sensor/camera/front_12mm/image/compressed",
                                                         CompressedImage,
                                                         qos_depth=10)

        self.actor.listen(self._sensor_data_updated)

    def _build_camera_info(self):
        """
        Private function to compute camera info

        camera info doesn't change over time
        """
        camera_info = CameraInfo()
        # store info without header
        camera_info.width = int(self.actor.attributes['image_size_x'])
        camera_info.height = int(self.actor.attributes['image_size_y'])
        camera_info.distortion_model = 'plumb_bob'
        cx = camera_info.width / 2.0
        cy = camera_info.height / 2.0
        fx = camera_info.width / (2.0 * math.tan(float(self.actor.attributes['fov']) * math.pi / 360.0))
        fy = fx
        camera_info.K.extend([fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0])
        camera_info.D.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        camera_info.R.extend([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        camera_info.P.extend([fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0])
        self._camera_info = camera_info

    def _get_image_data_array(self, carla_camera_data):
        """
        Function to transform the received carla camera data into a numpy data array

        The RGB camera provides a 4-channel int8 color format (bgra).
        """
        if ((carla_camera_data.height != self._camera_info.height) or
                (carla_camera_data.width != self._camera_info.width)):
            self.log.error("Camera{} received image not matching configuration".format(self.actor))
        carla_image_data_array = np.ndarray(shape=(carla_camera_data.height, carla_camera_data.width, 4),
                                            dtype=np.uint8,
                                            buffer=carla_camera_data.raw_data)

        return carla_image_data_array

    def get_msg_header(self, frame_id=None, timestamp=None):
        header = Header()
        if frame_id:
            header.frame_id = frame_id
        else:
            header.frame_id = "rgb"

        if not timestamp:
            timestamp = self.node.get_time()
        header.timestamp_sec = timestamp
        return header

    def _sensor_data_updated(self, carla_camera_data):
        """
        Function (override) to transform the received carla camera data
        into a Cyber image message
        """
        image_data_array = self._get_image_data_array(carla_camera_data)

        cam_compressed_img = CompressedImage()
        cam_compressed_img.header.CopyFrom(self.get_msg_header())
        cam_compressed_img.format = 'jpeg'
        cam_compressed_img.measurement_time = cam_compressed_img.header.timestamp_sec
        cam_compressed_img.data = cv2.imencode('.jpg', image_data_array)[1].tostring()

        # if "rgb-jiUdxPTthFYam9khom9tnW" == role_name:  # 236
        # if "rgb-2wkTERsLK9HByohVYKhqfg" == role_name:   # 159
        # if "rgb-PsuN5YNJ4ckohAC6HwQZG6" == self.role_name:   # 217
        if self.name == "front_6mm":
            cam_compressed_img.header.frame_id = "front_6mm"
            cam_compressed_img.frame_id = cam_compressed_img.header.frame_id
            self.compressed_6mm_writer.write(cam_compressed_img)

        # if "rgb-AWDfuPe2JdKae6ipPu85j5" == role_name:   # 236
        # if "rgb-iHyrAFDnMR6iPbtzurs3jV" == role_name:   # 159
        # if "rgb-D7crvKWaageLMN9ZCGuPa8" == self.role_name:   # 217
        if self.name == "front_12mm":
            cam_compressed_img.header.frame_id = "front_12mm"
            cam_compressed_img.frame_id = cam_compressed_img.header.frame_id
            self.compressed_12mm_writer.write(cam_compressed_img)

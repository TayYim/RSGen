#!/usr/bin/env python

# Copyright (c) 2023 synkrotron.ai.

""" Response to ETCD"""
import json
import time


class ResponseToEtcd:
    """push etcd信息"""

    def __init__(self, model, node, etcd):
        self.node = node
        self.model = model
        self.etcd_client = etcd
        self.lease = 300  # 租约时间

    def status(self, task_id, msg):
        """
        上报开始/结束
        """
        info = json.dumps(
            {
                "model": self.model,
                "status": msg,
                "now": time.strftime("%Y-%m-%d %H:%M:%S"),
                "ip": self.node,
            }
        )
        self.etcd_client.write(
            f"/oasis/tasks/{task_id}/{self.model}/status", info, ttl=self.lease
        )

    def exception(self, task_id, msg):
        """
        上报异常信息
        """
        info = json.dumps(
            {
                "model": self.model,
                "msg": msg,
                "now": time.strftime("%Y-%m-%d %H:%M:%S"),
                "ip": self.node,
            }
        )
        self.etcd_client.write(
            f"/oasis/tasks/{task_id}/exception", info, ttl=self.lease
        )

    def log(self, task_id, msg, log_type="log"):
        """
        上报日志信息
        """
        info = {
            "now": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": self.model,
            "msg": msg,
            "ip": self.node,
        }
        etcd_key = f"/oasis/tasks/{task_id}/{log_type}"
        if log_type == "status":
            info["status"] = msg
            etcd_key = f"/oasis/tasks/{task_id}/{self.model}/{log_type}"
        log_info = json.dumps(info)
        self.etcd_client.write(etcd_key, log_info, ttl=self.lease)

#!/usr/bin/env python
#
# SYNKROTRON Confidential
# Copyright (C) 2023 SYNKROTRON Inc. All rights reserved.
# The source code for this program is not published
# and protected by copyright controlled
#
"""redis utils"""

import redis
from utils.global_args import GlobalArgs


class RedisUtil:
    """redis util"""

    def __init__(self, host, port, password, db):
        self.host = host
        self.port = port
        self.password = password
        self.db = db
        self.log = GlobalArgs.log
        self.pool = None

    def __session(self):
        while True:
            try:
                if not self.pool:
                    self.create_pool()
                session = redis.Redis(connection_pool=self.pool)
                session.ping()
            except Exception:  # pylint: disable=W0718
                self.log.exception("redis connection failed, try to re-connecting!")
                continue
            else:
                return session

    def create_pool(self):
        self.pool = redis.ConnectionPool(
            host=str(self.host),
            port=int(self.port),
            password=str(self.password),
            db=int(self.db),
            decode_responses=True,
        )

    def get_redis(self):
        return self.__session()

    def lpush(self, key, value):
        return self.__session().lpush(key, value)

    def lpop(self, key):
        return self.__session().lpop(key)

    def get_task_use(self, key):
        return self.__session().get(key)

    def check_exist(self, key):
        return self.__session().exists(key)

    def get_init(self, key):
        return self.__session().get(key)

    def delete(self, key):
        return self.__session().delete(key)

    def bzpopmin(self, key, timeout=1):
        return self.__session().bzpopmin(key, timeout)

    def hget(self, name, key):
        return self.__session().hget(name, key)

    def set_task_use(self, key, value):
        self.__session().set(key, value)

    def hset(self, name, mapping_dict):
        return self.__session().hset(name, mapping=mapping_dict)

    def pubsub(self):
        return self.__session().pubsub()

    @staticmethod
    def subscribe(pubsub, channel):
        return pubsub.subscribe(channel)

    @staticmethod
    def listen(pubsub):
        return pubsub.listen()

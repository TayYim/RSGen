#!/bin/bash
pkill mainboard

# clear logs to save disk space
rm /apollo/data/log/*
rm /apollo/data/core/*
rm /apollo/nohup.out

ps -ef | grep run_osg.py | grep -v 'grep' | awk '{print $2}' | xargs kill -9
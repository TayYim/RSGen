#!/bin/bash
# Download oasis_bridge from the Internet or update
#git clone https://codeup.aliyun.com/5f3f374f6207a1a8b17f933f/stand_alone/apollo-bridge /apollo/modules/oasis_bridge
cd /apollo/modules/apollo-bridge

# Copy map files to /apollo/modules/map/data
cp -r map/. /apollo/modules/map/data

# Install requirements
pip3 install -r requirements.txt

# Set environment variables
if [ -f ~/.bashrc ] && ! grep -q 'export PYTHONPATH=$PYTHONPATH:/apollo/cyber' ~/.bashrc; then
    echo 'export PYTHONPATH=$PYTHONPATH:/apollo/cyber' >> ~/.bashrc
    echo 'export PYTHONPATH=$PYTHONPATH:/apollo/cyber/python' >> ~/.bashrc
    echo 'export PYTHONPATH=$PYTHONPATH:/apollo' >> ~/.bashrc
    echo 'export PYTHONPATH=$PYTHONPATH:/apollo/modules' >> ~/.bashrc
    echo 'export PYTHONPATH=$PYTHONPATH:/apollo/modules/apollo-bridge' >> ~/.bashrc
    echo 'export PYTHONPATH=$PYTHONPATH:/apollo/modules/apollo-bridge/carla_api/carla-0.9.14-py3.6-linux-x86_64.egg' >> ~/.bashrc
    echo 'export PYTHONPATH=$PYTHONPATH:/apollo/bazel-bin' >> ~/.bashrc
fi
source ~/.bashrc



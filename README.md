# RSGen
RSGen: A Tool for  Automated Generation of Risk-Stratified Traffic Scenarios (with Apollo 9.0 &amp; Carla integration)

Video link: [RSGen screencast @ FSE Demo 2025](https://www.youtube.com/watch?v=An7axjIgfaQ)

## System requirements

Ubuntu 20.04 and conda are necessary.

This tool is developed and tested on Ubuntu 20.04 with Python 3.10. Ubuntu 18.04 and 22.04 should work, but these versions were not tested. To run RSGen with Apollo and Carla, ensure your PC has at least 10GB of GPU memory. Our experiments were conducted on a PC with the following specifications: CPU: i7-13700KF, GPU: RTX 4070Ti, and 32GB RAM.

## Dependencies

### Carla

Install Carla 0.9.15. Download from https://github.com/carla-simulator/carla/releases.

Follow the doc: https://carla.readthedocs.io/en/latest/start_introduction/ .

Make sure the Carla simulator run normally with NVIDIA GPU. Running with CPU is not expected.

### Scenario execution framwork

We need Calra leaderboard and Scenario Runner as the scenario execution framwork framework.

```sh
git clone git@codeup.aliyun.com:5f3f374f6207a1a8b17f933f/TayYim/leaderboard.git -b osg

git clone git@codeup.aliyun.com:5f3f374f6207a1a8b17f933f/TayYim/scenario_runner.git -b osg
```

Install dependecies:
```sh
conda create -n lb python=3.7 -y \
&& conda activate lb

# in both leaderboard and scenario_runner
pip install -r requirements.txt
```

### Highway-env

Follow the doc of https://github.com/Farama-Foundation/HighwayEnv .

```sh
git clone https://github.com/TayYim/HighwayEnv.git

cd HighwayEnv

pip install -e .
```

## Setup Apollo

### Apollo 9.0

```sh
git clone https://github.com/TayYim/apollo.git -b r9_osg
```

Follow the guidence of [the official document](https://apollo.baidu.com/docs/apollo/9.x/md_docs_2_xE5_xAE_x89_xE8_xA3_x85_xE6_x8C_x87_xE5_x8D_x97_2_xE5_x8C_x85_xE7_xAE_xA1_xE7_x90_x86_410bb1324792103828eeacd86377c551.html) to to make Apollo docker.
Install nvidia docker as below:
```sh
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
&& curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
&& curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list \
&& sudo apt-get -y update \
&& sudo apt-get install -y nvidia-docker2
```
After this, restart docker:
```sh
sudo systemctl restart docker
```


Some extra changes is as below.

modules/planning/planning_base/gflags/planning_gflags.cc L21
```c
DEFINE_int32(planning_loop_rate, 100, "Loop rate for planning node");
```

modules/planning/planning_component/conf/planning.conf
```sh
--planning_upper_speed_limit=40.00
--default_cruise_speed=20.00
```

modules/planning/planning_component/conf/traffic_rule_config.pb.txt
```sh
rule {
  name: "TRAFFIC_LIGHT"
  type: "TrafficLight"
  enabled: false
}
```

There are some bugs in co-simulation between Apollo and Carla. See changes in https://github.com/PhilWallace/apollo-8.0.0_control_opt . It fixed some control bugs.

Then pull Apollo docker:
```sh
bash docker/scripts/dev_start.sh
```
Enter the docker:
```sh
bash docker/scripts/dev_into.sh
```
Compile runtime:
```sh
./apollo.sh build_opt_gpu
```


### Apollo 8.0

```
git clone https://github.com/ApolloAuto/apollo.git -b r8.0.0 apollo_8
```

Follow the official maunal to install.

To avoid duplicate of docker name, modify dev_start.sh & dev_into.sh 
```
DEV_CONTAINER_PREFIX='apollo_dev_8_'
```

Extra dependencies:

```
sudo apt install python3-pip -y
```

Also modify planning_gflags.cc, planning.conf and traffic_rule_config.pb.txt as we do in 9.0.

### Apollo 7.0

```sh
git clone https://github.com/ApolloAuto/apollo.git -b r7.0.0 apollo_7
```

Similar to 8.0. To avoid some zlib version issues during compiling, modify WORKSPACE L20:
```py
http_archive(
    name = "zlib",
    build_file = "@com_google_protobuf//:third_party/zlib.BUILD",
    sha256 = "629380c90a77b964d896ed37163f5c3a34f6e6d897311f1df2a7016355c45eff",
    strip_prefix = "zlib-1.2.11",
    urls = ["https://github.com/madler/zlib/archive/v1.2.11.tar.gz"],
)
```



## Setup Test Engine
Clone the repo:
```sh
git clone https://github.com/TayYim/RSGen.git
```

Create new conda env and install dependecies:
```sh
conda create -n rsgen python=3.10 -y \
&& conda activate rsgen \
&& pip install -r requirements.txt

# for server
pip install opencv-python-headless
```

### Configurations

Check `scripts/config.json`, make sure all paths are correct:
```py
  "env": {
    "CARLA_ROOT": "/home/you/CARLA",
    "SCENARIO_RUNNER_ROOT": "/home/you/Workspace/OSG/scenario_runner",
    "LEADERBOARD_ROOT": "/home/you/Workspace/OSG/leaderboard",
    "LB_PYTHON_PATH": "/home/you/miniconda3/envs/rsgen/bin/python",
    "RSGEN_ROOT": "/home/you/Workspace/RSGen"
  }
```

Add project path to python path:
```sh
echo 'export PYTHONPATH=$PYTHONPATH:YOUR_RSGEN_ROOT' >> ~/.bashrc
```


### Carla Apollo Bridge
Move apollo-bridge to apollo/modules.
```sh
# In Apollo container
./install.sh
```

## Run full experiments

To replicate the experiments described in the paper, simulations must be run across various $\omega$ values and scenarios. Use the `scripts/search/run.py` script to automate these simulations, as below:
```py
ws = [0.1, 0.3, 0.5, 0.7, 0.9]
for s in scenarios: # For each scenario
    for w in ws: # For each omega (risk level)
        main(name=s, w=w, agent="apollo", method="spso", output_base="output-fullrun")
```

On typical PCs, the script executes simulations sequentially due to high memory requirements. For faster processing, distribute simulations across multiple PCs with fixed random seeds.

The output data is stored at the path specified by the `output_base` parameter. It includes checkpoint files for resuming unfinished simulations, a logbook of intermediate optimizer values, a species file with final test suite values, and search tables summarizing simulation results.

More detailed explaination please refer to the Appendix of out paper.
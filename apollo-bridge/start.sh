#!/bin/bash

usage() {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  -h, --help        Show helo"
  echo "  -x, --dest_x      Dest x"
  echo "  -y, --dest_y      Dest y"
  echo "  -z, --dest_z      Dest z"
  echo "  -yaw, --dest_yaw  Dest yaw"
  echo "  -m, --map         Town name"
  exit 1
}

# set variable identifying the chroot you work in (used in the prompt below)
if [ -z "${debian_chroot:-}" ] && [ -r /etc/debian_chroot ]; then
    debian_chroot=$(cat /etc/debian_chroot)
fi


# colored GCC warnings and errors
#export GCC_COLORS='error=01;31:warning=01;35:note=01;36:caret=01;32:locus=01:quote=01'

# some more ls aliases
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'

if [ -f ~/.bash_aliases ]; then
    . ~/.bash_aliases
fi

# enable programmable completion features (you don't need to enable
# this, if it's already enabled in /etc/bash.bashrc and /etc/profile
# sources /etc/bash.bashrc).
if ! shopt -oq posix; then
  if [ -f /usr/share/bash-completion/bash_completion ]; then
    . /usr/share/bash-completion/bash_completion
  elif [ -f /etc/bash_completion ]; then
    . /etc/bash_completion
  fi
fi

########### source /opt/apollo/neo/setup.sh START ############

function pathremove() {
  local IFS=':'
  local NEWPATH
  local DIR
  local PATHVARIABLE=${2:-PATH}
  for DIR in ${!PATHVARIABLE}; do
    if [ "$DIR" != "$1" ]; then
      NEWPATH=${NEWPATH:+$NEWPATH:}$DIR
    fi
  done
  export $PATHVARIABLE="$NEWPATH"
}

function pathprepend() {
  pathremove $1 $2
  local PATHVARIABLE=${2:-PATH}
  export $PATHVARIABLE="$1${!PATHVARIABLE:+:${!PATHVARIABLE}}"
}

function pathappend() {
  pathremove $1 $2
  local PATHVARIABLE=${2:-PATH}
  export $PATHVARIABLE="${!PATHVARIABLE:+${!PATHVARIABLE}:}$1"
}

function setup_gpu_support() {
  if [ -e /usr/local/cuda/ ]; then
    pathprepend /usr/local/cuda/bin
  fi
}

if [ ! -f /apollo/LICENSE ]; then

  APOLLO_IN_DOCKER=false
  APOLLO_PATH="/opt/apollo/neo"
  APOLLO_ROOT_DIR=${APOLLO_PATH}/packages
  
  if [ -f /.dockerenv ]; then
    APOLLO_IN_DOCKER=true
  fi

  export APOLLO_PATH
  export APOLLO_ROOT_DIR=${APOLLO_PATH}/packages
  export CYBER_PATH=${APOLLO_ROOT_DIR}/cyber
  export APOLLO_IN_DOCKER
  export APOLLO_SYSROOT_DIR=/opt/apollo/sysroot
  export CYBER_DOMAIN_ID=80
  export CYBER_IP=127.0.0.1
  export GLOG_log_dir=${APOLLO_PATH}/data/log
  export GLOG_alsologtostderr=0
  export GLOG_colorlogtostderr=1
  export GLOG_minloglevel=0
  export sysmo_start=0
  export USE_ESD_CAN=false

fi


pathprepend /opt/apollo/neo/bin
setup_gpu_support

########### source /opt/apollo/neo/setup.sh END ############

export PYTHONPATH=$PYTHONPATH:/apollo/cyber
export PYTHONPATH=$PYTHONPATH:/apollo/cyber/python
export PYTHONPATH=$PYTHONPATH:/apollo
export PYTHONPATH=$PYTHONPATH:/apollo/modules
export PYTHONPATH=$PYTHONPATH:/apollo/modules/apollo-bridge
export PYTHONPATH=$PYTHONPATH:/apollo/modules/apollo-bridge/carla_api/carla-0.9.14-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:/apollo/bazel-bin


# Parse the command line parameters
parse_params() {
  # If no arguments are passed, show usage
  if [ $# -eq 0 ]; then
    usage
  fi

  # Reccursively parse the command line parameters
  while [ "$1" != "" ]; do
    case $1 in
      -h | --help)
        usage
        ;;
      -x | --dest_x)
        shift
        x=$1
        ;;
      -y | --dest_y)
        shift
        y=$1
        ;;
      -z | --dest_z)
        shift
        z=$1
        ;;
      -yaw | --dest_yaw)
        shift
        yaw=$1
        ;;
      -m | --map)
        shift
        map=$1
        ;;
      *)
        echo "Error, unknown param $1"
        usage
        ;;
    esac
    shift
  done
}

main() {
  parse_params "$@"

  cd /apollo/modules/apollo-bridge && python run_osg.py -x $x -y $y -z $z -yaw $yaw -m $map
}

main "$@"
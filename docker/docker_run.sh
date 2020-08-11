#!/bin/bash
#
# Usage:  ./docker_run.sh [/path/to/data]
#
# This script calls `docker` with `nvidia-docker2 runtime` to start the 
# labelfusion container with an interactive bash session.  This script 
# sets the required environment variables and mounts the labelfusion 
# source directory as a volume in the docker container.  If the path 
# to a data directory is given then the data directory is also mounted 
# as a volume.
#

#source ./config.sh
source_dir=$(cd $(dirname $0)/.. && pwd)

data_dir=/media/emo/Misc2/detection_data
data_mount_arg="-v $data_dir:/root/detection/data"

if [ ! -z "$1" ]; then

  data_dir=$1
  if [ ! -d "$data_dir" ]; then
    echo "directory does not exist: $data_dir"
    exit 1
  fi

  data_mount_arg="-v $data_dir:/root/detection/data"
fi

xhost +
docker run -it \
  --runtime=nvidia \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $source_dir:/root/detection $data_mount_arg \
  -v /dev/bus/usb:/dev/bus/usb \
  detection:0303 \
  bash
xhost -

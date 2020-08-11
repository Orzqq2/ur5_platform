#!/bin/bash
set -e

# setup ros environment
source /opt/ros/melodic/setup.bash
source /root/ws_emo/devel/setup.bash
exec "$@"

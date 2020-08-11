#!/bin/bash
#
# This script runs docker build to create the comac docker image.
#

set -exu   # http://linuxcommand.org/lc3_man_pages/seth.html

tag_name=ur5_platform_tensorflow:dev

cd ..

docker build -f docker/Docker1file --tag ${tag_name} .

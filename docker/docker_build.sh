#!/bin/bash
#
# This script runs docker build to create the comac docker image.
#

set -exu   # http://linuxcommand.org/lc3_man_pages/seth.html

tag_name=ur5_platform:dev

cd ..

docker build -f docker/Dockerfile --tag ${tag_name} .

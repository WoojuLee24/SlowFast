#!/bin/bash

docker run --rm -ti --runtime=nvidia --volume=$(pwd):/slowfast --workdir=/slowfast --ipc=host dnwn24/slowfast:pytorch1.5-cuda10.1-cudnn7-devel

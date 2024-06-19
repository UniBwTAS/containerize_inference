#!/usr/bin/env bash

DOCKER_IMAGE=$1
DOCKER_CONTAINER=$2
NEURAL_NETWORK_CONFIG=$3
DOCKER_HOST=$4
IPC_METHOD=$5
COMPRESSION=$6
NVIDIA_VISIBLE_DEVICES=$7

# get parent directory of this script in order to get path to ./script and ./data directories and Dockerfile
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/..
HOST_SCRIPTS_PATH="./scripts"
HOST_DATA_PATH="./data"

# get free port to use for ipc between ROS node and docker container for sensor data and inference results
PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

# if we run the docker on a remote machine then we have to copy ./scripts folder to that machine
if [ "$DOCKER_HOST" != "local" ]; then
  # it is not possible to use shared memory with remote docker
  IPC_METHOD="tcp_socket"

  # get ssh destination (e.g.: ssh://user@ip-address -> anre@ip-address)
  DOCKER_HOST_SSH=${DOCKER_HOST#"ssh://"}  # remove ssh:// prefix

  # get free port on remote machine to use for ipc between ROS node and docker container for sensor data and inference results
  PORT=$(ssh $DOCKER_HOST_SSH "python3 -c 'import socket; s=socket.socket(); s.bind((\"\", 0)); print(s.getsockname()[1]); s.close()'")

  # copy ./scripts folder to /tmp of remote machine (create individual folder for each individual container to avoid race conditions on code changes)
  ssh $DOCKER_HOST_SSH "mkdir -p /tmp/${DOCKER_CONTAINER}"
  scp -r $HOST_SCRIPTS_PATH $DOCKER_HOST_SSH:/tmp/${DOCKER_CONTAINER}

  # set the scripts path accordingly
  HOST_SCRIPTS_PATH="/tmp/${DOCKER_CONTAINER}/scripts"

  # share ./data folder between containers such that it is not necessary to download weights with each new container
  HOST_DATA_PATH="/tmp/containerize_inference_data"
fi

# docker build -f docker/$DOCKER_IMAGE.Dockerfile -t $DOCKER_IMAGE .
docker run \
  --rm \
  --runtime=nvidia \
  -p $PORT:$PORT \
  -v $HOST_SCRIPTS_PATH:/scripts \
  -v $HOST_DATA_PATH:/data \
  -v /dev/shm:/dev/shm \
  -e PYTHONUNBUFFERED=1 \
  -e NVIDIA_VISIBLE_DEVICES=$NVIDIA_VISIBLE_DEVICES \
  --name $DOCKER_CONTAINER \
  $DOCKER_IMAGE \
  python3 /scripts/inference_docker.py $DOCKER_IMAGE $DOCKER_CONTAINER $NEURAL_NETWORK_CONFIG $DOCKER_HOST $IPC_METHOD $PORT $COMPRESSION
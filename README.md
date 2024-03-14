# Dockerize your Deep Neural Networks (DNN) and use them in ROS1

- 🚀 Get rid of dependency (CUDA, PyTorch, cuDNN, ...) problems and install them in docker container
- 🚀 Simple ROS agnostic communication (IPC) between Docker container and ROS1 node
  - IPC based on Shared Memory (very low latency, 2ms)
  - IPC based on TCP socket (allows inference on remote machine with strong GPUs)
- 🚀 Automatic deployment: build, start and stop of Docker container

## Install:

1. [Install Docker](https://docs.docker.com/engine/install/ubuntu/)
2. [Install NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-with-apt)
3. [Install ROS1](http://wiki.ros.org/noetic/Installation/Ubuntu)
4. [Setup ROS workspace](https://catkin-tools.readthedocs.io/en/latest/quick_start.html#initializing-a-new-workspace) (if not already existing)
5. Navigate to `src` folder of your catkin workspace (or some sub-folder of your choice)
6. Get GitHub Dependencies & this project:
```shell
# ROS message definitions for instance/semantic/panoptic segmentation & object detection
git clone https://github.com/UniBwTAS/object_instance_msgs.git

# RVIZ plugin to visualize above messages
git clone https://github.com/UniBwTAS/rviz_object_instance.git

# our ROS package
git clone https://github.com/UniBwTAS/containerize_inference.git
```
7. Get some other dependencies via rosdep:
```shell
sudo rosdep update
sudo rosdep install --from-paths . --ignore-src -r -y
```
8. Build everything
9. Run node (with YOLOv8) + visualization:
```shell
roslaunch containerize_inference inference.launch
```
> [!NOTE]
> The first time a new `docker_image` is used this takes some time as it has to be built first! In subsequent calls this will be much faster.

## Available Docker Images (and Models) so far:
- [mmdetection](https://github.com/open-mmlab/mmdetection/tree/main/configs):
  - `mask-rcnn_r50_fpn_1x_coco` (Instance Segmentation)
  - `mask2former_swin-t-p4-w7-224_8xb2-lsj-50e_coco-panoptic` (Panoptic Segmentation, Tiny)
  - `mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco-panoptic` (Panoptic Segmentation, Small)
  - `mask2former_swin-l-p4-w12-384-in21k_16xb1-lsj-100e_coco-panoptic` (Panoptic Segmentation, Large)
  - `faster-rcnn_r50_fpn_1x_coco` (Object Detection)
  - and more... (see GitHub project)
- [mmdetection3d](https://github.com/open-mmlab/mmdetection3d/tree/main/configs)
  - (Not fully implemented yet)
- [ultralytics](https://github.com/ultralytics/ultralytics) (YOLOv8)
  - `yolov8m.pt` (Object Detection)
  - `yolov8m-seg.pt` (Instance Segmentation)
  - and more... (see GitHub project)

```shell
roslaunch containerize_inference inference.launch docker_image:=ultralytics neural_network_config:=yolov8m-seg.pt
```

## Run inference on remote machine:
```shell
roslaunch containerize_inference inference.launch docker_host:="ssh://anre@137.193.76.4"
```
> [!NOTE]
> Docker and NVIDIA Container Toolkit has to be installed on remote machine.

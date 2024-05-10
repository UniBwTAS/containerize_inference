#!/usr/bin/env python

import numpy as np
import rospy
import sys
import time
from cv_bridge import CvBridge
from object_instance_msgs.msg import ObjectInstance2DArray, ObjectInstance3DArray
from sensor_msgs.msg import Image, CompressedImage, PointCloud2
from std_msgs.msg import Header

from utils.dataset import COCO_PANOPTIC_CLASSES
from utils.docker_utils import get_ip_for_docker_container, get_port_for_docker_container
from utils.ipc_utils import SharedMemoryCommunication, TcpSocketCommunication
from utils.ros_utils import numpy_to_object_instance_2d_array_msg, point_cloud2_msg_to_numpy, is_compressed_image_topic, \
    ros_image_encoding_to_cv_encoding


class InferenceHost:
    def __init__(self, ipc, input_type, output_type, compression="none"):
        self.compression = compression
        self.ipc = ipc

        # init opencv bridge
        self.bridge = CvBridge()

        # init subscriber
        if input_type == "image":
            input_msg_type = CompressedImage if is_compressed_image_topic("input") else Image
            self.sub = rospy.Subscriber("input", input_msg_type, self.image_callback, queue_size=1, tcp_nodelay=True)
        elif input_type == "point_cloud":
            self.sub = rospy.Subscriber("input", PointCloud2, self.point_cloud_callback, queue_size=1, tcp_nodelay=True)

        # init publisher
        if output_type == "instances_2d":
            self.pub = rospy.Publisher("instances_2d", ObjectInstance2DArray, queue_size=1)
        elif output_type == "instances_3d":
            self.pub = rospy.Publisher("instances_3d", ObjectInstance3DArray, queue_size=1)
        elif output_type == "semantic_segmentation" and input_type == "point_cloud":
            self.pub = rospy.Publisher("semantic_segmentation", PointCloud2, queue_size=1)
        self.pub_finished = rospy.Publisher("inference_complete_trigger", Header, queue_size=1)  # lightweight msg

    def image_callback(self, msg):
        # discard if input latency is already too high (full queue)
        input_delay = (rospy.Time.now() - msg.header.stamp).to_sec()
        if input_delay > 0.1:
            self.pub_finished.publish(msg.header)
            print("[HOST] Discard message with latency:", input_delay)
            return
        print("[HOST] Input latency", input_delay)

        # convert image message to numpy array
        start_time_to_cv = time.time()
        if type(msg) == CompressedImage:
            numpy_array = np.ndarray(shape=(len(msg.data),), dtype=np.uint8, buffer=msg.data)
            compression = msg.format
        else:
            numpy_array = self.bridge.imgmsg_to_cv2(msg, "passthrough")
            compression = "none"
            if self.compression != "none" and not msg.encoding.startswith("bayer_"):
                # numpy_array is compressed automatically during send
                compression = self.compression
        print("[HOST] Duration [to_cv]:", time.time() - start_time_to_cv)

        # send data to docker container
        start_time_send_img = time.time()
        input_encoding = "BGR" if type(msg) == CompressedImage else ros_image_encoding_to_cv_encoding(msg.encoding)
        self.ipc.start_send()
        self.ipc.write_opencv_image(numpy_array, input_encoding, compression)
        self.ipc.flush()

        # receive data from docker container
        self.ipc.receive()
        instance_segmentation = self.ipc.read_opencv_image()
        semantic_segmentation = self.ipc.read_opencv_image()
        instance_ids = self.ipc.read_numpy_array()
        labels = self.ipc.read_numpy_array()
        scores = self.ipc.read_numpy_array()
        boxes = self.ipc.read_numpy_array()
        print("[HOST] Duration [send_img, inference, receive_inference]:", time.time() - start_time_send_img)

        # convert numpy arrays to ROS message
        start_time_to_msg = time.time()
        msg_out = numpy_to_object_instance_2d_array_msg(msg.header, instance_segmentation, semantic_segmentation,
                                                        instance_ids, labels, scores, boxes, COCO_PANOPTIC_CLASSES)
        print("[HOST] Duration [to_msg]:", time.time() - start_time_to_msg)

        # publish ROS message
        start_time_pub = time.time()
        self.pub.publish(msg_out)
        self.pub_finished.publish(msg.header)
        print("[HOST] Duration [pub]:", time.time() - start_time_pub)
        print("[HOST] Duration [total]:", time.time() - start_time_to_cv)
        print("[HOST] Output latency:", (rospy.Time.now() - msg.header.stamp).to_sec())

    def point_cloud_callback(self, point_cloud):
        # discard if input latency is already too high (full queue)
        input_delay = (rospy.Time.now() - point_cloud.header.stamp).to_sec()
        if input_delay > 0.15:
            print("[HOST] Discard message with latency:", input_delay)
            return
        print("[HOST] Input latency", input_delay)

        # point cloud to numpy array
        start_time_to_msg = time.time()
        pc = point_cloud2_msg_to_numpy(point_cloud)
        print("[HOST] Duration [to_msg]:", time.time() - start_time_to_msg)

        # send data to docker container
        start_time_send = time.time()
        self.ipc.start_send()
        self.ipc.write_numpy_array(pc)
        self.ipc.flush()
        print("[HOST] Duration [send]:", time.time() - start_time_send)

        # receive data from docker container
        start_time_receive = time.time()
        self.ipc.receive()
        labels = self.ipc.read_numpy_array()
        scores = self.ipc.read_numpy_array()
        boxes_3d = self.ipc.read_numpy_array()
        print("[HOST] Duration [receive]:", time.time() - start_time_receive)

        # convert numpy arrays to ROS message
        start_time_to_msg = time.time()
        # msg = numpy_to_object_instance_2d_array_msg(msg.header, instance_segmentation, semantic_segmentation, labels,
        #                                             scores, boxes, COCO_PANOPTIC_CLASSES)
        print("[HOST] Duration [to_msg]:", time.time() - start_time_to_msg)

        # publish ROS message
        # start_time_pub = time.time()
        # self.pub.publish(msg)
        # print("[HOST] Duration [pub]:", time.time() - start_time_pub)
        # print("[HOST] Output latency:", (rospy.Time.now() - msg.header.stamp).to_sec())


def main():
    docker_image = sys.argv[1]
    docker_container = sys.argv[2]
    config = sys.argv[3]
    docker_host = sys.argv[4]
    ipc_method = sys.argv[5]
    compression = sys.argv[6]

    ipc = None
    try:
        # setup connection to docker container
        if docker_host != "local" or ipc_method == "tcp_socket":
            if docker_host == "local":
                ip_address = get_ip_for_docker_container(docker_container)
            else:
                ip_address = docker_host.split("@")[-1]
            port = get_port_for_docker_container(docker_container)

            ipc = TcpSocketCommunication(server_mode=False, print_prefix="HOST")
            ipc.connect((ip_address, port))
        elif ipc_method == "shared_memory":
            ipc = SharedMemoryCommunication(server_mode=False, print_prefix="HOST")
            ipc.connect(("/" + docker_container, "/" + docker_container))

        # get input/output types
        input_type = "image"
        output_type = "instances_2d"
        if docker_image == "mmdetection3d":
            if config.startswith("centerpoint"):
                input_type = "point_cloud"
                output_type = "instances_3d"
            elif config.startswith("cylinder3d"):
                input_type = "point_cloud"
                output_type = "semantic_segmentation"

        # run node
        rospy.init_node('inference')
        node = InferenceHost(ipc, input_type, output_type, compression)
        rospy.spin()
    except KeyboardInterrupt:
        print("[HOST] Inference stopped by user...")
    finally:
        if ipc is not None:
            # close communication
            ipc.close()


if __name__ == '__main__':
    main()

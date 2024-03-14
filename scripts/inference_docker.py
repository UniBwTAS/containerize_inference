#!/usr/bin/env python

import numpy as np
import sys
import time
import torch

from utils.ipc_utils import SharedMemoryCommunication, TcpSocketCommunication
from utils.weight_utils import download_weights, download_config_and_weights_with_mim


class Inference:
    def __init__(self, config, compression="none", print_prefix="DOCKER"):
        self.config = config
        self.compression = compression
        self.print_prefix = print_prefix

    def data_received_callback(self, arrays):
        raise NotImplementedError("")

    def run(self, ipc):
        while True:
            ipc.receive()
            self.data_received_callback(ipc)


class MMDetectionInference(Inference):
    def __init__(self, config, compression="none", print_prefix="DOCKER"):
        super().__init__(config, compression, print_prefix)

        self.config_file, self.weights_file = download_config_and_weights_with_mim(self.config, "/data",
                                                                                   print_prefix=print_prefix)
        if self.config_file is None:
            raise RuntimeError("Unable to download the weights!")
        from mmdet.apis import init_detector
        self.model = init_detector(self.config_file, self.weights_file, device='cuda:0')

    def data_received_callback(self, ipc):
        start_time_total = time.time()

        # get image from inter process communication (and convert encoding to BGR if needed)
        img = ipc.read_opencv_image(desired_output_encoding='BGR')

        # inference on given image
        start_time_inference = time.time()
        from mmdet.apis import inference_detector
        result = inference_detector(self.model, img)
        print(f"[{self.print_prefix}] Duration [inference]:", time.time() - start_time_inference)

        # get result tensors
        instance_segmentation = None
        semantic_segmentation = None
        if hasattr(result, "pred_panoptic_seg"):
            semantic_segmentation = (result.pred_panoptic_seg.sem_seg % 1000).type(
                torch.int16).cpu().numpy()[0, ...]
            semantic_segmentation = semantic_segmentation.astype(np.uint16)
        if hasattr(result.pred_instances, "masks"):
            # fuse masks for each instance and bring to same format as panoptic segmentation tensor
            d = result.pred_instances.masks.device
            r = torch.arange(1, result.pred_instances.masks.shape[0] + 1, dtype=torch.int16, device=d)
            instance_segmentation = (result.pred_instances.masks * r[:, None, None]).max(0).values.cpu().numpy()
            instance_segmentation = instance_segmentation.astype(np.uint16)
        instance_ids = None
        labels = result.pred_instances.labels.cpu().numpy()
        scores = result.pred_instances.scores.cpu().numpy()
        boxes = result.pred_instances.bboxes.cpu().numpy()
        print(f"[{self.print_prefix}] Duration [total]:", time.time() - start_time_total)

        # send the data to host
        compression = "png" if self.compression != "none" else "none"
        ipc.start_send()
        ipc.write_opencv_image(instance_segmentation, "GRAY", compression)
        ipc.write_opencv_image(semantic_segmentation, "GRAY", compression)
        ipc.write_numpy_array(instance_ids)
        ipc.write_numpy_array(labels)
        ipc.write_numpy_array(scores)
        ipc.write_numpy_array(boxes)
        ipc.flush()


class MMDetection3DInference(Inference):
    def __init__(self, config, compression, print_prefix="DOCKER"):
        super().__init__(config, compression, print_prefix)

        self.config_file, self.weights_file = download_config_and_weights_with_mim(config, "/data", "mmdet3d",
                                                                                   print_prefix=print_prefix)
        if self.config_file is None:
            raise RuntimeError("Unable to download the weights!")

        from mmdet3d.apis import init_model
        self.model = init_model(self.config_file, self.weights_file, device='cuda:0')

    def data_received_callback(self, ipc):
        # get point cloud from inter process communication
        pc = ipc.read_numpy_array()

        # add another column
        pc = np.concatenate((pc, np.empty((pc.shape[0], 1), dtype=pc.dtype)), axis=1)

        # inference on given image
        start_time_inference = time.time()
        from mmdet3d.apis import inference_detector
        result = inference_detector(self.model, pc)
        # result = inference_segmentor(self.model, pc)
        print(f"[{self.print_prefix}] Duration [inference]:", time.time() - start_time_inference)

        # get result tensors
        labels = result[0].pred_instances_3d.labels_3d.cpu().numpy()
        scores = result[0].pred_instances_3d.scores_3d.cpu().numpy()
        boxes_3d = result[0].pred_instances_3d.bboxes_3d.cpu().numpy()

        # send the data to host
        ipc.start_send()
        ipc.write_numpy_array(labels)
        ipc.write_numpy_array(scores)
        ipc.write_numpy_array(boxes_3d)
        ipc.flush()


class UltralyticsInference(Inference):
    def __init__(self, config, compression, print_prefix="DOCKER"):
        super().__init__(config, compression, print_prefix)

        self.weights = download_weights("https://github.com/ultralytics/assets/releases/download/v8.1.0/" + self.config,
                                        "/data", print_prefix)
        if self.weights is None:
            raise RuntimeError("Unable to download the weights!")

        from ultralytics import YOLO
        self.model = YOLO("/data/" + self.config)
        self.model.fuse()

    def data_received_callback(self, ipc):
        start_time_total = time.time()

        # get image from inter process communication (and convert encoding to BGR if needed)
        img = ipc.read_opencv_image(desired_output_encoding='BGR')

        # inference on given image
        start_time_inference = time.time()
        results = self.model.track(source=img, verbose=False, tracker="bytetrack.yaml", retina_masks=True)
        print(f"[{self.print_prefix}] Duration [inference]:", time.time() - start_time_inference)

        # get result tensors
        result = results[0]
        instance_segmentation = None
        semantic_segmentation = None
        instance_ids = result.boxes.id if result.boxes.id is not None else torch.arange(result.boxes.cls.shape[0])
        if hasattr(result, "masks") and result.masks is not None and result.boxes.id is not None:
            instance_segmentation = (
                    result.masks.data * instance_ids.to(result.masks.data.device)[:, None, None]).max(
                dim=0).values.cpu().detach().numpy().astype(np.uint16)
        instance_ids = instance_ids.cpu().numpy()
        labels = result.boxes.cls.cpu().detach().numpy()
        scores = result.boxes.conf.cpu().detach().numpy()
        boxes = result.boxes.xyxy.cpu().detach().numpy()
        print(f"[{self.print_prefix}] Duration [total]:", time.time() - start_time_total)

        # send the data to host
        compression = "png" if self.compression != "none" else "none"
        ipc.start_send()
        ipc.write_opencv_image(instance_segmentation, "GRAY", compression)
        ipc.write_opencv_image(semantic_segmentation, "GRAY", compression)
        ipc.write_numpy_array(instance_ids)
        ipc.write_numpy_array(labels)
        ipc.write_numpy_array(scores)
        ipc.write_numpy_array(boxes)
        ipc.flush()


def main():
    docker_image = sys.argv[1]
    docker_container = sys.argv[2]
    config = sys.argv[3]
    docker_host = sys.argv[4]
    ipc_method = sys.argv[5]
    port = int(sys.argv[6])
    compression = sys.argv[7]

    inference = None
    if docker_image == "mmdetection":
        inference = MMDetectionInference(config, compression)
    elif docker_image == "mmdetection3d":
        inference = MMDetection3DInference(config, compression)
    elif docker_image == "ultralytics":
        inference = UltralyticsInference(config, compression)

    ipc = None
    try:
        if docker_host != "local" or ipc_method == "tcp_socket":
            ipc = TcpSocketCommunication(server_mode=True, print_prefix="DOCKER")
            ipc.connect((port,))
        elif ipc_method == "shared_memory":
            ipc = SharedMemoryCommunication(server_mode=True, print_prefix="DOCKER")
            ipc.connect(("/" + docker_container, "/" + docker_container))
        inference.run(ipc)
    except KeyboardInterrupt:
        print("[DOCKER] Inference stopped by user...")
    finally:
        if ipc is not None:
            ipc.close()


if __name__ == '__main__':
    main()

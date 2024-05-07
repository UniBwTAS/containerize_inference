import numpy as np
import rospy.names
import rostopic
from object_instance_msgs.msg import ObjectInstance2DArray, ObjectInstance2D
from ros_numpy import numpify


def point_cloud2_msg_to_numpy(msg):
    data = numpify(msg)
    intensity_key = 'intensity' if 'intensity' in data.dtype.names else 'i'
    data = np.stack(
        (data['x'].flatten(), data['y'].flatten(), data['z'].flatten(), data[intensity_key].flatten()),
        axis=1)  # TODO: need to scale intensity value for kitti, waymo, tas, etc. ?
    return data[~np.isnan(data).any(axis=1), :]


def numpy_to_object_instance_2d_array_msg(header, instance_segmentation, semantic_segmentation, instance_ids, labels,
                                          scores, boxes, full_classes_list):
    msg = ObjectInstance2DArray()
    msg.header = header
    i = 0

    if labels is not None and scores is not None and boxes is not None:
        for label, score, box in zip(labels, scores, boxes):
            instance_msg = ObjectInstance2D()
            instance_msg.id = i + 1 if instance_ids is None else int(instance_ids[i])
            instance_msg.is_instance = True
            instance_msg.class_name = full_classes_list[int(label)]
            instance_msg.class_index = int(label)
            instance_msg.class_probabilities = [float(score), ]
            instance_msg.class_count = len(full_classes_list)
            instance_msg.bounding_box_min_x = int(box[0])
            instance_msg.bounding_box_min_y = int(box[1])
            instance_msg.bounding_box_max_x = int(box[2])
            instance_msg.bounding_box_max_y = int(box[3])
            msg.instances.append(instance_msg)
            i += 1

    # instance part of panoptic segmentation
    if instance_segmentation is not None:
        msg.instance_mask.header = header
        msg.instance_mask.height = instance_segmentation.shape[0]
        msg.instance_mask.width = instance_segmentation.shape[1]
        msg.instance_mask.encoding = "mono16"
        msg.instance_mask.step = 2 * msg.instance_mask.width
        msg.instance_mask.data = instance_segmentation.tobytes()

        # semantic part of panoptic segmentation
    if semantic_segmentation is not None:
        msg.semantic_mask.header = header
        msg.semantic_mask.height = semantic_segmentation.shape[0]
        msg.semantic_mask.width = semantic_segmentation.shape[1]
        msg.semantic_mask.encoding = "mono16"
        msg.semantic_mask.step = 2 * msg.semantic_mask.width
        msg.semantic_mask.data = semantic_segmentation.tobytes()
        msg.semantic_class_indices = [i for i in range(len(full_classes_list) + 1)]
        msg.semantic_class_names = full_classes_list + ("unknown",)

    return msg


def ros_image_encoding_to_cv_encoding(encoding):
    # analogous to http://docs.ros.org/en/jade/api/sensor_msgs/html/image__encodings_8h_source.html
    # however the output strings are compliant to opencv's cvtColor conversion codes, i.e. "COLOR_" + a + "2" + b
    if encoding == "mono8" or encoding == "mono16":
        return "GRAY"
    if encoding == "bgr8" or encoding == "bgr16":
        return "BGR"
    if encoding == "rgb8" or encoding == "rgb16":
        return "RGB"
    if encoding == "bgra8" or encoding == "bgra16":
        return "BGRA"
    if encoding == "rgba8" or encoding == "rgba16":
        return "RGBA"
    if encoding == "yuv422":
        return "YUV"
    if encoding == "bayer_rggb8" or encoding == "bayer_rggb16":
        return "BayerRGGB"
    if encoding == "bayer_bggr8" or encoding == "bayer_bggr16":
        return "BayerBGGR"
    if encoding == "bayer_gbrg8" or encoding == "bayer_gbrg16":
        return "BayerGBRG"
    if encoding == "bayer_grbg8" or encoding == "bayer_grbg16":
        return "BayerGRBG"
    raise RuntimeError("Unsupported image encoding: " + encoding)


def is_compressed_image_topic(topic):
    return rostopic.get_topic_type(rospy.names.resolve_name(topic), blocking=True) == 'sensor_msgs/CompressedImage'

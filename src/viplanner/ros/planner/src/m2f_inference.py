# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# python
import numpy as np

# ROS
import rospy
from mmdet.apis import inference_detector, init_detector
from mmdet.evaluation import INSTANCE_OFFSET

# viplanner-ros
from viplanner.config.coco_sem_meta import get_class_for_id_mmdet
from viplanner.config.viplanner_sem_meta import VIPlannerSemMetaHandler


class Mask2FormerInference:
    """Run Inference on Mask2Former model to estimate semantic segmentation"""

    debug: bool = False

    def __init__(
        self,
        config_file="configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml",
        checkpoint_file="model_final.pth",
    ) -> None:

        # Build the model from a config file and a checkpoint file
        # Wrap init in a try/except to provide a clearer error message when
        # mmengine tries to read configs from an unreadable installed path
        # (PermissionError seen when mmdetection was installed from a root-owned
        # source tree). If that occurs, instruct the user to provide an
        # accessible config path or fix permissions / reinstall mmdetection.
        try:
            config_file = (
                "/home/doge/models/mmdetection/configs/"
                "mask2former/mask2former_r50_8xb2-lsj-50e_coco-panoptic.py"
            )

            checkpoint_file = (
                "/home/doge/models/mask2former/"
                "mask2former_r50_8xb2-lsj-50e_coco-panoptic_20230118_125535-54df384a.pth"
            )

            self.model = init_detector(config_file, checkpoint_file, device="cuda:0")
        except PermissionError as e:
            # Provide a more actionable error message than the raw traceback
            msg = (
                "Permission denied while loading mmdetection config. "
                "mmengine attempted to open a config file in an installation "
                "directory (e.g. '/root/git/mmdetection/...') that is not readable.\n"
                "Solutions:\n"
                "  1) Pass an absolute, readable config file path to Mask2FormerInference, "
                "for example a config bundled with this repo or one you installed.\n"
                "  2) Fix permissions on the installed mmdetection repo (e.g. `sudo chmod -R a+r /path/to/mmdetection/configs`)\n"
                "  3) Reinstall mmdetection as your user (avoid root-owned source trees).\n"
                "Original error: " + str(e)
            )
            rospy.logerr(msg)
            # Re-raise with the clearer message so the node fails fast and the
            # user sees how to fix it.
            raise RuntimeError(msg) from e

        # mapping from coco class id to viplanner class id and color
        viplanner_meta = VIPlannerSemMetaHandler()
        coco_viplanner_cls_mapping = get_class_for_id_mmdet(self.model.dataset_meta["classes"])
        self.viplanner_sem_class_color_map = viplanner_meta.class_color
        self.coco_viplanner_color_mapping = {}
        for coco_id, viplanner_cls_name in coco_viplanner_cls_mapping.items():
            self.coco_viplanner_color_mapping[coco_id] = viplanner_meta.class_color[viplanner_cls_name]

        return

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Predict semantic segmentation from image

        Args:
            image (np.ndarray): image to be processed in BGR format
        """

        result = inference_detector(self.model, image)
        result = result.pred_panoptic_seg.sem_seg.detach().cpu().numpy()[0]
        # create output
        panoptic_mask = np.zeros((result.shape[0], result.shape[1], 3), dtype=np.uint8)
        for curr_sem_class in np.unique(result):
            curr_label = curr_sem_class % INSTANCE_OFFSET
            try:
                panoptic_mask[result == curr_sem_class] = self.coco_viplanner_color_mapping[curr_label]
            except KeyError:
                if curr_sem_class != len(self.model.dataset_meta["classes"]):
                    rospy.logwarn(f"Category {curr_label} not found in" " coco_viplanner_cls_mapping.")
                panoptic_mask[result == curr_sem_class] = self.viplanner_sem_class_color_map["static"]

        if self.debug:
            import matplotlib.pyplot as plt

            plt.imshow(panoptic_mask)
            plt.show()

        return panoptic_mask


# EoF

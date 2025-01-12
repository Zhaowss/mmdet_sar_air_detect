# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

from torch import Tensor
from torch_geometric.loader import DataLoader
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .base import BaseDetector
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import torch
from sklearn.cluster import KMeans
from .GCNdataloader import create_graph_data
from .GCNATTEN import  GraphNet
from .Gatattention import EnhancedGraphNetWithGAT
from torch_geometric.data import Batch
from .EGNFF import  EnhancedGraphNetWithFeatureFusion
@MODELS.register_module()
class SingleStageDetector(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = MODELS.build(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.GCN=GraphNet(input_dim=2,hidden_dim=512,output_dim=100).cuda()
        self.GCNAt=EnhancedGraphNetWithGAT(input_dim=2,hidden_dim=512,output_dim=100).cuda()
        self.GATweight=EnhancedGraphNetWithFeatureFusion(hidden_dim=512,num_fpn_layers=3)

    def _load_from_state_dict(self, state_dict: dict, prefix: str,
                              local_metadata: dict, strict: bool,
                              missing_keys: Union[List[str], str],
                              unexpected_keys: Union[List[str], str],
                              error_msgs: Union[List[str], str]) -> None:
        """Exchange bbox_head key to rpn_head key when loading two-stage
        weights into single-stage model."""
        bbox_head_prefix = prefix + '.bbox_head' if prefix else 'bbox_head'
        bbox_head_keys = [
            k for k in state_dict.keys() if k.startswith(bbox_head_prefix)
        ]
        rpn_head_prefix = prefix + '.rpn_head' if prefix else 'rpn_head'
        rpn_head_keys = [
            k for k in state_dict.keys() if k.startswith(rpn_head_prefix)
        ]
        if len(bbox_head_keys) == 0 and len(rpn_head_keys) != 0:
            for rpn_head_key in rpn_head_keys:
                bbox_head_key = bbox_head_prefix + \
                                rpn_head_key[len(rpn_head_prefix):]
                state_dict[bbox_head_key] = state_dict.pop(rpn_head_key)
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        x ,graph_exact_feature = self.extract_feat(batch_inputs)
        losses = self.bbox_head.loss(x, batch_data_samples,graph_exact_feature)
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        x, graph_exact_feature= self.extract_feat(batch_inputs)
        results_list = self.bbox_head.predict(
            x, batch_data_samples,graph_exact_feature, rescale=rescale)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

         Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.

        Returns:
            tuple[list]: A tuple of features from ``bbox_head`` forward.
        """
        x ,graph_exact_feature= self.extract_feat(batch_inputs)
        results = self.bbox_head.forward(x)
        return results

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        # 图特征的提取
        all_corner_tensors, graph_features=self.extract_graph_features(batch_inputs)
        graph_data_list=create_graph_data(all_corner_tensors,graph_features)
        batch_input_feature=Batch.from_data_list(graph_data_list)
        graph_exact_feature=self.GCNAt(batch_input_feature)
        x = self.backbone(batch_inputs)

        x=self.GATweight(graph_exact_feature,x)
        if self.with_neck:
            x = self.neck(x)
        return x,graph_exact_feature

    def extract_graph_features(self, img):
        """Extract corner points and construct graph features."""
        # List to store corner tensors
        all_corner_tensors = []
        # Maximum number of corners to consider
        max_num_corners = 100

        # Loop through each image
        for image in img:
            # Convert to numpy format
            image = image.permute(1, 2, 0).cpu().numpy()
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            # Detect corners using GoodFeaturesToTrack
            corners = cv2.goodFeaturesToTrack(
                gray_image, maxCorners=128, qualityLevel=0.20, minDistance=10
            )

            if corners is None:
                print("No corners detected, skipping this image.")
                continue

            corners = corners.reshape(-1, 2)
            num_corners = len(corners)

            if num_corners > max_num_corners:
                # Sort corners by intensity and keep the top `max_num_corners`
                corner_values = [gray_image[int(y), int(x)] for x, y in corners]
                sorted_corners = sorted(zip(corners, corner_values), key=lambda x: x[1], reverse=True)
                corners = np.array([corner for corner, _ in sorted_corners[:max_num_corners]])
            else:
                # Duplicate corners to match `max_num_corners` if needed
                num_repeats = max_num_corners - num_corners
                repeated_corners = np.tile(corners, (num_repeats // num_corners, 1))
                remainder = num_repeats % num_corners
                if remainder > 0:
                    repeated_corners = np.vstack((repeated_corners, corners[:remainder]))
                corners = np.vstack((corners, repeated_corners))

            # Add corners to the list
            all_corner_tensors.append(corners)

        # Convert list of corner tensors to a single tensor
        all_corner_tensors = torch.tensor(np.array(all_corner_tensors), dtype=torch.float32).to('cuda:0')

        # Compute pairwise similarity to construct graph features
        num_images, num_corners, _ = all_corner_tensors.shape
        graph_features = torch.zeros((num_images, max_num_corners, max_num_corners), dtype=torch.float32).to('cuda:0')

        for i in range(num_images):
            corners = all_corner_tensors[i]
            pairwise_distances = torch.cdist(corners, corners,2)  # Compute pairwise distances
            graph_features[i] = torch.exp(-pairwise_distances)  # Similarity based on distance

        return all_corner_tensors, graph_features

# Copyright (C) 2023 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import numpy as np
import cv2
import torch
import os
from typing import Optional
from segment_anything import sam_model_registry, SamPredictor

def index_of_same_image(list_of_images, image):
    for i, cached_img in enumerate(list_of_images):
        is_the_same_image = cached_img is not None and np.array_equal(np.array(image), cached_img)
        if is_the_same_image:
            return i
    return None

def convert_mask_to_polygon(mask):
    contours = None
    if int(cv2.__version__.split('.')[0]) > 3:
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)[0]
    else:
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)[1]

    contours = max(contours, key=lambda arr: arr.size)
    if contours.shape.count(1):
        contours = np.squeeze(contours)
    if contours.size < 3 * 2:
        raise Exception('Less then three point have been detected. Can not build a polygon.')

    polygon = []
    for point in contours:
        polygon.append([int(point[0]), int(point[1])])

    return polygon

class ModelHandler:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sam_checkpoint = "/opt/nuclio/sam/sam_vit_b_01ec64.pth"
        self.model_type = "vit_b"
        self.amount_of_images = int(os.environ.get('AMOUNT_OF_IMAGES_TO_CACHE', '10'))
        self.latest_images = [np.array([]) for _ in range(self.amount_of_images)]
        self.latest_low_res_masks : list[Optional[np.ndarray]] = [None] * self.amount_of_images
        sam_model = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        sam_model.to(device=self.device)
        self.sam_predictors = [SamPredictor(sam_model) for _ in range(self.amount_of_images)]
        self.last_image_idx = 0

    def handle(self, image, pos_points, neg_points):
        # latest 'amount_of_images' images are kept in memory because function is always run-time after startup
        # we use to avoid computing emeddings twice for the cached images
        idx = index_of_same_image(self.latest_images, image)
        if idx is None:
            idx = self.last_image_idx
            self.latest_low_res_masks[idx] = None
            numpy_image = np.array(image)
            self.sam_predictors[idx].set_image(numpy_image)
            self.latest_images[idx] = numpy_image
            self.last_image_idx = (self.last_image_idx + 1) % self.amount_of_images
        # we assume that pos_points and neg_points are of type:
        # np.array[[x, y], [x, y], ...]
        input_points = np.array(pos_points)
        input_labels = np.array([1] * len(pos_points))

        if len(neg_points):
            input_points = np.concatenate([input_points, neg_points], axis=0)
            input_labels = np.concatenate([input_labels, np.array([0] * len(neg_points))], axis=0)

        masks, _, low_res_masks = self.sam_predictors[idx].predict(
            point_coords=input_points,
            point_labels=input_labels,
            mask_input = self.latest_low_res_masks[idx],
            multimask_output=False
        )
        self.latest_low_res_masks[idx] = low_res_masks
        object_mask = np.array(masks[0], dtype=np.uint8)
        cv2.normalize(object_mask, object_mask, 0, 255, cv2.NORM_MINMAX)
        polygon = convert_mask_to_polygon(object_mask)
        return object_mask, polygon

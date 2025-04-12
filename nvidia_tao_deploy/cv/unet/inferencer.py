# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility class for performing TensorRT image inference."""

import os
import cv2
from PIL import Image
import numpy as np

import tensorrt as trt

from nvidia_tao_deploy.cv.unet.utils import get_color_id, overlay_seg_image
from nvidia_tao_deploy.inferencer.trt_inferencer import TRTInferencer
from nvidia_tao_deploy.inferencer.utils import allocate_buffers, do_inference


def trt_output_process_fn(y_encoded, model_output_width, model_output_height, activation="softmax"):
    """Function to process TRT model output."""
    predictions_batch = []
    for idx in range(y_encoded[0].shape[0]):
        pred = np.reshape(y_encoded[0][idx, ...], (model_output_height,
                                                   model_output_width,
                                                   1))
        pred = np.squeeze(pred, axis=-1)
        if activation == "sigmoid":
            pred = np.where(pred > 0.5, 1, 0)
        pred = pred.astype(np.int32)
        predictions_batch.append(pred)
    return np.array(predictions_batch)


class UNetInferencer(TRTInferencer):
    """Manages TensorRT objects for model inference."""
    def __init__(
        self,
        engine_path,
        input_shape=None,
        batch_size=None,
        data_format="channel_first",
        activation="softmax",
    ):
        """Initializes TensorRT objects needed for model inference.

        Args:
            engine_path (str): path where TensorRT engine should be stored
            input_shape (tuple): (batch, channel, height, width) for dynamic shape engine
            batch_size (int): batch size for dynamic shape engine
            data_format (str): either channel_first or channel_last
        """

        super().__init__(engine_path)
        self.activation = activation
        self.context = self.engine.create_execution_context()

        # Get input tensor name and shape
        self.input_tensor_name = None
        self._input_shape = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_tensor_name = name
                self._input_shape = list(self.engine.get_tensor_shape(name))
                break

        assert self.input_tensor_name is not None, "No input tensor found in the engine"
        assert len(self._input_shape) == 4, "Expected shape (N,C,H,W) or similar"

        # Handle batch size or dynamic shapes
        if input_shape is not None:
            self.context.set_input_shape(self.input_tensor_name, input_shape)
            self.max_batch_size = input_shape[0]
        elif batch_size is not None:
            dynamic_shape = [batch_size] + self._input_shape[1:]
            self.context.set_input_shape(self.input_tensor_name, dynamic_shape)
            self.max_batch_size = batch_size
        else:
            self.max_batch_size = self._input_shape[0]  # May be -1 for dynamic

        # Set width and height based on data_format
        if data_format == "channel_first":
            self.height = self._input_shape[2]
            self.width = self._input_shape[3]
        else:
            self.height = self._input_shape[1]
            self.width = self._input_shape[2]

        # Allocate buffers using new TRT 10 API
        self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(
            self.engine, self.context
        )

        # Preallocate numpy buffer
        input_volume = trt.volume(self._input_shape[1:])  # exclude batch
        self.numpy_array = np.zeros(
            (self.max_batch_size, input_volume), dtype=np.float32
        )

    def infer(self, imgs):
        """Infers model on batch of same sized images resized to fit the model.

        Args:
            image_paths (str): paths to images, that will be packed into batch
                and fed into model
        """
        # Verify if the supplied batch size is not too big
        max_batch_size = self.max_batch_size
        actual_batch_size = len(imgs)
        if actual_batch_size > max_batch_size:
            raise ValueError(f"image_paths list bigger ({actual_batch_size}) than \
                               engine max batch size ({max_batch_size})")

        self.numpy_array[:actual_batch_size] = imgs.reshape(actual_batch_size, -1)
        # ...copy them into appropriate place into memory...
        # (self.inputs was returned earlier by allocate_buffers())
        np.copyto(self.inputs[0].host, self.numpy_array.ravel())

        # ...fetch model outputs...
        results = do_inference(
            self.context, bindings=self.bindings, inputs=self.inputs,
            outputs=self.outputs, stream=self.stream,
            batch_size=max_batch_size,
            execute_v2=self.execute_v2)

        # ...and return results up to the actual batch size.
        y_pred = [i.reshape(max_batch_size, -1)[:actual_batch_size] for i in results]

        # Process TRT outputs to proper format
        return trt_output_process_fn(y_pred, self.width, self.height, self.activation)

    def __del__(self):
        """Clear things up on object deletion."""
        # Clear session and buffer
        if self.trt_runtime:
            del self.trt_runtime

        if self.context:
            del self.context

        if self.engine:
            del self.engine

        if self.stream:
            del self.stream

        # Loop through inputs and free inputs.
        for inp in self.inputs:
            inp.device.free()

        # Loop through outputs and free them.
        for out in self.outputs:
            out.device.free()

    def visualize_masks(self, img_paths, predictions, out_dir, num_classes=2,
                        input_image_type="rgb", resize_padding=False, resize_method='BILINEAR'):
        """Store overlaid image and predictions to png format.

        Args:
            img_paths: The input image names.
            predictions: Predicted masks numpy arrays.
            out_dir: Output dir where the visualization is saved.
            num_classes: Number of classes used.
            input_image_type: The input type of image (color/ grayscale).
            resize_padding: If padding was used or not.
            resize_method: Resize method used (Default: BILINEAR).
        """
        colors = get_color_id(num_classes)

        vis_dir = os.path.join(out_dir, "vis_overlay")
        label_dir = os.path.join(out_dir, "mask_labels")
        os.makedirs(vis_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)

        for pred, img_path in zip(predictions, img_paths):
            segmented_img = np.zeros((self.height, self.width, 3))
            pred = self.remove_small_regions(pred, class_id=1, min_area=5000)
            img_file_name = os.path.basename(img_path)
            for c in range(len(colors)):
                seg_arr_c = pred[:, :] == c
                segmented_img[:, :, 0] += ((seg_arr_c) * (colors[c][0])).astype('uint8')
                segmented_img[:, :, 1] += ((seg_arr_c) * (colors[c][1])).astype('uint8')
                segmented_img[:, :, 2] += ((seg_arr_c) * (colors[c][2])).astype('uint8')

            orig_image = cv2.imread(img_path)

            if input_image_type == "grayscale":
                pred = pred.astype(np.uint8) * 255
                fused_img = Image.fromarray(pred).resize(size=(self.width, self.height),
                                                         resample=Image.BILINEAR)
                # Save overlaid image
                fused_img.save(os.path.join(vis_dir, img_file_name))
            else:
                segmented_img = np.zeros((self.height, self.width, 3))
                for c in range(len(colors)):
                    seg_arr_c = pred[:, :] == c
                    segmented_img[:, :, 0] += ((seg_arr_c) * (colors[c][0])).astype('uint8')
                    segmented_img[:, :, 1] += ((seg_arr_c) * (colors[c][1])).astype('uint8')
                    segmented_img[:, :, 2] += ((seg_arr_c) * (colors[c][2])).astype('uint8')
                orig_image = cv2.imread(img_path)
                fused_img = overlay_seg_image(orig_image, segmented_img, resize_padding,
                                              resize_method)
                # Save overlaid image
                cv2.imwrite(os.path.join(vis_dir, img_file_name), fused_img)
            mask_name = f"{os.path.splitext(img_file_name)[0]}.png"

            # Save predictions
            cv2.imwrite(os.path.join(label_dir, mask_name), pred)

    def remove_small_regions(self, mask, class_id=1, min_area=2000):
        """Remove small connected components of the specified class."""
        binary_mask = (mask == class_id).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8
        )
        for i in range(0, num_labels):  # Skip background
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                mask[labels == i] = 0  # Replace with background
        return mask
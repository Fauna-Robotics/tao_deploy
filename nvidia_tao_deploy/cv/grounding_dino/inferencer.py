# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""TensorRT Engine class for Deformable DETR."""

from nvidia_tao_deploy.inferencer.trt_inferencer import TRTInferencer
from nvidia_tao_deploy.inferencer.utils import allocate_buffers, do_inference
import numpy as np
from PIL import ImageDraw

import tensorrt as trt  # pylint: disable=unused-import


def trt_output_process_fn(y_encoded, batch_size, num_classes):
    """Function to process TRT model output.

    Args:
        y_encoded (list): list of TRT outputs in numpy
        batch_size (int): batch size from TRT engine
        num_classes (int): number of classes that the model was trained on

    Returns:
        pred_logits (np.ndarray): (B x NQ x N) logits of the prediction
        pred_boxes (np.ndarray): (B x NQ x 4) bounding boxes of the prediction
    """
    pred_logits, pred_boxes = y_encoded
    return pred_logits.reshape((batch_size, -1, num_classes)), pred_boxes.reshape((batch_size, -1, 4))


class GDINOInferencer(TRTInferencer):
    """Implements inference for the G-DINO TensorRT engine."""

    def __init__(self, engine_path, num_classes, input_shape=None, batch_size=None, data_format="channel_first"):
        """Initializes TensorRT objects needed for model inference.

        Args:
            engine_path (str): path where TensorRT engine should be stored
            num_classes (int): number of classes that the model was trained on
            input_shape (tuple): (batch, channel, height, width) for dynamic shape engine
            batch_size (int): batch size for dynamic shape engine
            data_format (str): either channel_first or channel_last
        """
        # Load TRT engine
        super().__init__(engine_path)

        # Find max shape
        binding_index = self.engine.get_binding_index(self.engine.get_binding_name(0))
        min_shape, opt_shape, max_shape = self.engine.get_profile_shape(0, binding_index)
        self.max_batch_size = max_shape[0]

        self.execute_v2 = False

        # Execution context is needed for inference
        self.context = None

        # Allocate memory for multiple usage [e.g. multiple batch inference]
        self._input_shape = []
        self.context = self.engine.create_execution_context()
        for binding in range(self.engine.num_bindings):
            # set binding_shape for dynamic input
            if self.engine.binding_is_input(binding):
                _input_shape = self.engine.get_binding_shape(binding)[1:]
                self._input_shape.append(_input_shape)
                self.context.set_binding_shape(binding, [batch_size] + list(_input_shape))
                if binding == 0 and len(_input_shape) == 3:
                    self.height = _input_shape[1]
                    self.width = _input_shape[2]
        self.max_batch_size = batch_size
        self.execute_v2 = True

        self.num_classes = num_classes

        # This allocates memory for network inputs/outputs on both CPU and GPU
        self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.engine,
                                                                                 self.context)
        if self.context is None:
            self.context = self.engine.create_execution_context()

        input_volumes = [trt.volume(shape) for shape in self._input_shape]
        dtypes = (float, int, bool, int, int, bool)
        self.numpy_array = [
            np.zeros((self.max_batch_size, volume), dtype=dtype) for volume, dtype in zip(input_volumes, dtypes)
        ]

    def infer(self, inputs):
        """Infers model on batch of same sized images resized to fit the model.

        Args:
            image_paths (str): paths to images, that will be packed into batch
                and fed into model
        """
        # Verify if the supplied batch size is not too big
        max_batch_size = self.max_batch_size
        for idx, inp in enumerate(inputs):
            actual_batch_size = len(inp)
            if actual_batch_size > max_batch_size:
                raise ValueError(
                    f"image_paths list bigger ({actual_batch_size}) than "
                    f"engine max batch size ({max_batch_size})"
                )
            self.numpy_array[idx][:actual_batch_size] = inp.reshape(actual_batch_size, -1)
            # ...copy them into appropriate place into memory...
            # (self.inputs was returned earlier by allocate_buffers())
            np.copyto(self.inputs[idx].host, self.numpy_array[idx].ravel())

        # ...fetch model outputs...
        results = do_inference(
            self.context, bindings=self.bindings, inputs=self.inputs,
            outputs=self.outputs, stream=self.stream,
            batch_size=max_batch_size,
            execute_v2=self.execute_v2)

        # ...and return results up to the actual batch size.
        y_pred = [i.reshape(max_batch_size, -1)[:actual_batch_size] for i in results]

        # Process TRT outputs to proper format
        results = trt_output_process_fn(y_pred, actual_batch_size, self.num_classes)
        return results

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

    def draw_bbox(self, img, prediction, class_mapping, threshold=0.3, color_map=None):  # noqa pylint: disable=W0237
        """Draws bbox on image and dump prediction in KITTI format

        Args:
            img (numpy.ndarray): Preprocessed image
            prediction (numpy.ndarray): (N x 6) predictions
            class_mapping (dict): key is the class index and value is the class name
            threshold (float): value to filter predictions
            color_map (dict): key is the class name and value is the color to be used
        """
        draw = ImageDraw.Draw(img)

        label_strings = []
        for i in prediction:
            if int(i[0]) not in class_mapping:
                print(i[0], class_mapping)
                continue
            cls_name = class_mapping[int(i[0])]
            if float(i[1]) < threshold:
                continue

            if cls_name in color_map:
                draw.rectangle(((i[2], i[3]), (i[4], i[5])),
                               outline=color_map[cls_name])
                # txt pad
                draw.rectangle(((i[2], i[3] - 10), (i[2] + (i[4] - i[2]), i[3])),
                               fill=color_map[cls_name])
                draw.text((i[2], i[3] - 10), f"{cls_name}: {i[1]:.2f}")

            x1, y1, x2, y2 = float(i[2]), float(i[3]), float(i[4]), float(i[5])
            label_head = cls_name + " 0.00 0 0.00 "
            bbox_string = f"{x1:.3f} {y1:.3f} {x2:.3f} {y2:.3f}"
            label_tail = f" 0.00 0.00 0.00 0.00 0.00 0.00 0.00 {float(i[1]):.3f}\n"
            label_string = label_head + bbox_string + label_tail
            label_strings.append(label_string)
        return img, label_strings

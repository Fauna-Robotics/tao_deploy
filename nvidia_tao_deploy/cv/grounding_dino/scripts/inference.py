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

"""Standalone TensorRT inference."""

import os
import logging
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from nvidia_tao_deploy.cv.centerpose.utils import (
    merge_outputs,
)
from nvidia_tao_deploy.cv.common.decorators import monitor_status
from nvidia_tao_deploy.cv.grounding_dino.inferencer import GDINOInferencer
from nvidia_tao_deploy.cv.grounding_dino.hydra_config.default_config import ExperimentConfig
from nvidia_tao_deploy.cv.grounding_dino.utils import post_process, tokenize_captions
from nvidia_tao_deploy.utils.image_batcher import ImageBatcher
from nvidia_tao_deploy.cv.common.hydra.hydra_runner import hydra_runner


logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)
spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "specs"),
    config_name="infer", schema=ExperimentConfig
)
@monitor_status(name='grounding_dino', mode='inference')
def main(cfg: ExperimentConfig) -> None:
    """Grounding DINO TRT Inference."""
    if not os.path.exists(cfg.inference.trt_engine):
        raise FileNotFoundError(f"Provided inference.trt_engine at {cfg.inference.trt_engine} does not exist!")

    max_text_len = cfg.model.max_text_len

    trt_infer = GDINOInferencer(cfg.inference.trt_engine,
                                batch_size=cfg.dataset.batch_size,
                                num_classes=max_text_len)

    _, c, h, w = trt_infer._input_shape[0]
    batcher = ImageBatcher(list(cfg.dataset.infer_data_sources.image_dir),
                           (cfg.dataset.batch_size, c, h, w),
                           trt_infer.inputs[0].host.dtype,
                           preprocessor="DDETR")

    cat_list = list(cfg.dataset.infer_data_sources["captions"])
    caption = [" . ".join(cat_list) + ' .'] * cfg.dataset.batch_size
    inv_classes = dict(enumerate(cat_list))

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_encoder_type)
    input_ids, attention_mask, position_ids, token_type_ids, text_self_attention_masks, pos_map = tokenize_captions(tokenizer, cat_list, caption, max_text_len)

    # Create results directories
    if cfg.inference.results_dir:
        results_dir = cfg.inference.results_dir
    else:
        results_dir = os.path.join(cfg.results_dir, "trt_inference")
    os.makedirs(results_dir, exist_ok=True)
    output_annotate_root = os.path.join(results_dir, "images_annotated")
    output_label_root = os.path.join(results_dir, "labels")

    os.makedirs(output_annotate_root, exist_ok=True)
    os.makedirs(output_label_root, exist_ok=True)

    for batches, img_paths, scales in tqdm(batcher.get_batch(), total=batcher.num_batches, desc="Producing predictions"):
        # Handle last batch as we artifically pad images for the last batch idx
        if len(img_paths) != len(batches):
            batches = batches[:len(img_paths)]
            input_ids = input_ids[:len(img_paths)]
            attention_mask = attention_mask[:len(img_paths)]
            position_ids = position_ids[:len(img_paths)]
            token_type_ids = token_type_ids[:len(img_paths)]
            text_self_attention_masks = text_self_attention_masks[:len(img_paths)]

        inputs = (batches, input_ids, attention_mask, position_ids, token_type_ids, text_self_attention_masks)
        pred_logits, pred_boxes = trt_infer.infer(inputs)

        pred_logits = pred_logits.reshape(1, cfg.model.num_queries, cfg.model.hidden_dim)
        pred_boxes = pred_boxes.reshape(1, cfg.model.num_queries, 4)

        target_sizes = []
        for batch, scale in zip(batches, scales):
            _, new_h, new_w = batch.shape
            orig_h, orig_w = int(scale[0] * new_h), int(scale[1] * new_w)
            target_sizes.append([orig_w, orig_h, orig_w, orig_h])

        class_labels, scores, boxes = post_process(pred_logits, pred_boxes, target_sizes, pos_map)
        y_pred_valid = merge_detections(scores, boxes, class_labels)

        for img_path, pred in zip(img_paths, y_pred_valid):
            # Load Image
            img = Image.open(img_path)

            # Resize of the original input image is not required for D-DETR
            # as the predictions are rescaled in post_process
            bbox_img, label_strings = trt_infer.draw_bbox(img, pred, inv_classes, cfg.inference.conf_threshold, cfg.inference.color_map)
            img_filename = os.path.basename(img_path)
            bbox_img.save(os.path.join(output_annotate_root, img_filename))

            # Store labels
            filename, _ = os.path.splitext(img_filename)
            label_file_name = os.path.join(output_label_root, filename + ".txt")
            with open(label_file_name, "w", encoding="utf-8") as f:
                for l_s in label_strings:
                    f.write(l_s)

    logging.info("Finished inference.")

def merge_detections(scores, boxes, class_labels):
    """
    Applies NMS to the detections.
    Note: This assumes batch size of 1.
    """
    # Build a dict for detections before applying NMS.
    detections = [
        [
            {
                "score": float(scores[0][i]),
                "bbox": boxes[0][i].tolist(),
                "label": int(class_labels[0][i]),
            }
            for i in range(len(scores[0]))
        ]
    ]
    merged_detections = merge_outputs(detections)
    merged = merged_detections[0]

    class_labels = np.array([d["label"] for d in merged])
    scores = np.array([d["score"] for d in merged])
    boxes = np.array([d["bbox"] for d in merged])

    # Ensure consistent shapes
    class_labels = np.atleast_1d(class_labels).reshape(-1, 1)
    scores = np.atleast_1d(scores).reshape(-1, 1)
    boxes = np.atleast_2d(boxes).reshape(-1, 4)

    # Concatenate to (N, 6)
    y_pred = np.concatenate([class_labels, scores, boxes], axis=-1)
    # Add batch dim to get (1, N, 6)
    y_pred_valid = y_pred[None, ...]

    return y_pred_valid
if __name__ == '__main__':
    main()

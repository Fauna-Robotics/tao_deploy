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

"""OpticalInpsection convert etlt model to TRT engine."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import tempfile

from nvidia_tao_deploy.cv.common.decorators import monitor_status
from nvidia_tao_deploy.cv.optical_inspection.engine_builder import OpticalInpsectionEngineBuilder
from nvidia_tao_deploy.cv.common.hydra.hydra_runner import hydra_runner
from nvidia_tao_deploy.cv.optical_inspection.config.default_config import (
    OIExperimentConfig as ExperimentConfig
)
from nvidia_tao_deploy.utils.decoding import decode_model

logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)
spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "specs"),
    config_name="experiment", schema=ExperimentConfig
)
@monitor_status(name="optical_inspection", mode="gen_trt_engine")
def main(cfg: ExperimentConfig) -> None:
    """Convert encrypted uff or onnx model to TRT engine."""
    # decrypt etlt
    trt_cfg = cfg.gen_trt_engine
    # decrypt onnx or etlt
    tmp_onnx_file, file_format = decode_model(trt_cfg.onnx_file, cfg['encryption_key'])

    engine_file = trt_cfg.trt_engine

    batch_size = trt_cfg.batch_size
    data_type = trt_cfg.tensorrt['data_type']
    workspace_size = trt_cfg.tensorrt['workspace_size']
    min_batch_size = trt_cfg.tensorrt['min_batch_size']
    opt_batch_size = trt_cfg.tensorrt['opt_batch_size']
    max_batch_size = trt_cfg.tensorrt['max_batch_size']

    if engine_file is None:
        engine_handle, temp_engine_path = tempfile.mkstemp()
        os.close(engine_handle)
        output_engine_path = temp_engine_path
    else:
        output_engine_path = engine_file

    builder = OpticalInpsectionEngineBuilder(
        verbose=trt_cfg.verbose,
        workspace=workspace_size,
        batch_size=batch_size,
        min_batch_size=min_batch_size,
        opt_batch_size=opt_batch_size,
        max_batch_size=max_batch_size)
    builder.create_network(tmp_onnx_file, file_format)
    builder.create_engine(
        output_engine_path,
        data_type)

    logging.info("Engine generation finished successfully.")


if __name__ == '__main__':
    main()

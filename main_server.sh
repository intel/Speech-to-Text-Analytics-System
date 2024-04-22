#!/bin/bash

# Copyright (C) 2023 Intel Corporation                                                                                              
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
#

#export KMP_BLOCKTIME=1
#export KMP_SETTINGS=1
#export KMP_AFFINITY=granularity=fine,compact,1,0
export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
export ONEDNN_PRIMITIVE_CACHE_CAPACITY=4096
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"

OMP_NUM_THREADS=56 numactl -C 0-55 serve run rest_api.application.api_ingress

# For maximum core utilization and pin to specific core uncomment following command. and comment above script command.
#OMP_NUM_THREADS=56 numactl -C 0-55 -m 0 python main.py --config_file ${CONFIG_FILE} --input_audio ${INPUT_FILE} --output_dir ${OUT_DIR}

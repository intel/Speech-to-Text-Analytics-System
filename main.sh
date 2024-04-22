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

export KMP_BLOCKTIME=1
export KMP_SETTINGS=1
export KMP_AFFINITY=granularity=fine,compact,1,0
export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
export ONEDNN_PRIMITIVE_CACHE_CAPACITY=4096
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"

CONFIG_FILE='./config/asr.yaml'
OUT_DIR="./logs/"

while [ "$1" != "" ]; 
do
   case $1 in
    -c | --config )
        shift
        CONFIG_FILE="$1"
        echo "Output Directory is : $CONFIG_FILE"
        ;;
    -i | --input )
        shift
        INPUT_FILE="$1"
        echo "Input file is : $INPUT_FILE"
        ;;
    -o | --output_dir )
        shift
        OUT_DIR="$1"
        echo "Output directory is : $OUT_DIR"
        ;;
    -g | --ground_truth )
        shift
        GROUND_TRUTH="$1"
        echo "Ground truth is : $GROUND_TRUTH"
        ;;
    -h | --help ) 
        echo "Usage: ./main.sh [OPTIONS]"
        echo "OPTION includes:"
        echo "   -c | --config - YAML Configuration file." 
        echo "   -i | --input - Input audio file."
        echo "   -o | --output_dir - Output directory where result wil be saved."
        echo "   -g | - ground_truth - Ground truth file to compute accuracy."
        echo "   -h | --help - displays this message"
        exit
      ;;
    * ) 
        echo "Invalid option: $1"
        echo "Usage: ./main.sh [OPTIONS]"
        echo "OPTION includes:"
        echo "   -c | --config - YAML Configuration file." 
        echo "   -i | --input - Input audio file."
        echo "   -o | --output_dir - Output directory where result wil be saved."
        echo "   -g | - ground_truth - Ground truth file to compute accuracy."
        echo "   -h | --help - displays this message"
        exit
       ;;
  esac
  shift
done

echo "COMMAND: python main.py --config_file $CONFIG_FILE --input_audio $INPUT_FILE --output_dir $OUT_DIR --ground_truth $GROUND_TRUTH"


python main.py --config_file ${CONFIG_FILE} --input_audio ${INPUT_FILE} --output_dir ${OUT_DIR}

# For maximum core utilization and pin to specific core uncomment following command. and comment above script command.
#OMP_NUM_THREADS=56 numactl -C 0-55 -m 0 python main.py --config_file ${CONFIG_FILE} --input_audio ${INPUT_FILE} --output_dir ${OUT_DIR}

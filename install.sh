#!/bin/bash

INSALL_MODE="bare_metal"

while [ "$1" != "" ]; 
do
   case $1 in
    -im | --install_mode )
        shift
        INSALL_MODE="$1"
        echo "Installation mode is $INSALL_MODE"
        ;;
  esac
  shift
done
echo "Installation mode is $INSALL_MODE"
if [ "$INSALL_MODE" != 'docker' ]; then
   conda install -c conda-forge ffmpeg -y
fi

#pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cpu
pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers==4.37.2
pip install python-dotenv
pip install datasets
python -m pip install optimum
pip install accelerate
#pip install pyannote.audio==3.0.1
pip install pyannote.audio
python -m pip install intel_extension_for_pytorch
pip install git+https://github.com/m-bain/whisperx.git
pip install pytest
pip install "ray[serve]"
pip install starlette==0.36.2
pip install fastapi==0.109.1
pip install streamlit==1.31.0
pip install markdown
pip install python-multipart

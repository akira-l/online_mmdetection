#!/bin/bash

# uncompress the packed enviroment into the local dir
function prepare_env() {

  #tar -xf "afs/env/pack_download.tar" 
  #pip install "pack_download/addict-2.3.0-py3-none-any.whl"
  #pip install "pack_download/numpy-1.19.2-cp37-cp37m-manylinux2010_x86_64.whl"
  #pip install "pack_download/opencv_python-4.4.0.44-cp37-cp37m-manylinux2014_x86_64.whl"
  #pip install "pack_download/yapf-0.30.0-py2.py3-none-any.whl"
  #pip install "pack_download/mmcv_full-latest+torch1.3.0+cu92-cp37-cp37m-manylinux1_x86_64.whl"  

  #unzip "afs/env/mmcv-master.zip"
  #cd mmcv-master
  #MMCV_WITH_OPS=1 pip install -e .  # package mmcv-full will be installed after this step
  #cd .. 

  tar -xf "afs/env/open-mmlab.tar"
  export PATH="$(pwd)/envs/open-mmlab/bin:$PATH"
  export PYTHONPATH="$(pwd)/envs/open-mmlab/bin:$PYTHONPATH"
  export LD_LIBRARY_PATH="$(pwd)/envs/open-mmlab/lib:$LD_LIBRARY_PATH"
  echo "$(pwd)"
  #export LD_LIBRARY_PATH="$(pwd)/env/usr/local/lib:$LD_LIBRARY_PATH"
  #export LD_LIBRARY_PATH="$(pwd)/env/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH"
  #export LD_LIBRARY_PATH="$(pwd)/env/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

}


function prepare_data() {
  mkdir -p data

  #cp afs/dataset/textvqa/wiki.en.bin ./
  #cp -r afs/dataset/textvqa/cache/torch/* /root/.cache/torch/

  START=`date +%s%N`;
  tar -xf "afs/dataset/coco2017.tar" 
  END=`date +%s%N`;
  time=$((END-START))
  time=`expr $time / 1000000000`
  echo "time for unzip afs/dataset/coco2017.tar" 
  echo $time
}

# unify ui
prepare_env
prepare_data
export PYTHONPATH="$(pwd)"

python env_test.py
#python tools/train.py configs/mask_rcnn/mask_rcnn_r50_caffe_c4_1x_coco.py
bash tools/dist_train.sh configs/mask_rcnn/mask_rcnn_r50_caffe_c4_1x_coco.py 4 

echo "files in current path"
work_path=$(pwd)
files=$(ls $work_path)
for filename in $files
do
   echo $filename
done

echo "files in afs" 
afs_path=$(pwd)/afs
files=$(ls $afs_path)
for filename in $files
do
   echo $filename
done



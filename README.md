# Chimera - NAS for Event-Based Object Detection

This repository contains the source code for "Chimera: A Block-Based Neural Architecture Search Framework for Event-Based Object Detection", accepted for publication in the Frontiers in Artificial Intelligence journal. <br/>
![GEN1 results](images/cover.png) <br/>
You can read the full paper on: <br/>
[Outdated Arxiv Paper - to be updated soon](https://arxiv.org/pdf/2412.19646?) <br/>

# Setting up the environment 

conda create -n reyolov8 python==3.9 <br/>
conda activate reyolov8 <br/>
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge <br/>
python -m pip install opencv-python tqdm==4.66.2 pandas==2.2.2 numpy==1.22.4 psutil==5.9.8 pyyaml==6.0.1 matplotlib==3.8.4 thop wandb h5py==3.11.0 hdf5plugin==4.4.0 tensorboard==2.16.2 <br/>


# Evaluate 

python val.py --data ${DATASET}.yaml --model ${WEIGHTS}.pt  --channels 5  <br/>


# Training 

# Single-gpu

 python train.py --batch 12 --nbs 6 --epochs 100 --data ${DATASET}.yaml  --model ${MODEL_NAME}.yaml --channels 5 --name ${WANDB_RUN_NAME} --project ${WANDB_PROJECT_NAME} <br/>



# TODO MULTI-GPU

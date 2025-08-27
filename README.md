# Chimera - NAS for Event-Based Object Detection

This repository contains the source code for "Chimera: A Block-Based Neural Architecture Search Framework for Event-Based Object Detection", accepted for publication in the Frontiers in Artificial Intelligence journal. <br/>
![GEN1 results](images/cover.png) <br/>
You can read the full paper on: <br/>
[Outdated Arxiv Paper - to be updated soon](https://arxiv.org/pdf/2412.19646?) <br/>

# Setting up the environment 

conda env create -f chimera_env.yml <br/>

# Evaluate 

python val.py --data ${DATASET}.yaml --model ${WEIGHTS}.pt  --channels 5  <br/>


# Training 

# Single-gpu

 python train.py --batch 12 --nbs 6 --epochs 100 --data ${DATASET}.yaml  --model ${MODEL_NAME}.yaml --channels 5 --name ${WANDB_RUN_NAME} --project ${WANDB_PROJECT_NAME} <br/>



# TODO MULTI-GPU

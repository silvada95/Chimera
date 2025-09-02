# Chimera - NAS for Event-Based Object Detection

This repository contains the source code for "Chimera: A Block-Based Neural Architecture Search Framework for Event-Based Object Detection", accepted for publication in the Frontiers in Artificial Intelligence journal. <br/>
![GEN1 results](images/cover.png) <br/>
You can read the full paper on: <br/>
[Outdated Arxiv Paper - to be updated soon](https://arxiv.org/pdf/2412.19646?) <br/>

# Setting up the environment 
```
conda env create -f chimera_env.yml 
```

# Executing Chimera-NAS

The general setup for Chimera-NAS is:
```
python ChimeraSearch.py --exp_name ${EXP_NAME} --population ${INDIVIDUALS} --iterations ${IT} --max_size ${MAX_PARAMS} --features ${LIST_OF_PROXIES} --w_macs ${W_MACS} --w_meco ${W_MECO} --w_zen ${W_ZEN} --w_azexpr ${W_AZEXPR} --w_aztrain ${W_AZTRAIN} --w_azprogr ${W_AZPROGR}
```
where:  <br />
**EXP_NAME**: Name of the folder where your run will be stored. Default: "chimera-nas" <br />
**INDIVIDUALS**: Number of individuals in your population. Default is 100. <br />
**IT**: Number of iterations on the evolutionary search. Default is 1000. <br />
**MAX_PARAMS**: Maximum number of parameters that an individual from the population can have. It is in the scale of millions already. (If you want 10M, just set it as 10) <br />
**LIST_OF_PROXIES**: An additive list of proxies that will be used by the search algorithm. The options are "macs", "meco", "zen score", "az-train", "az-prog", "az-expr". <br />
**W_MACS**, **W_MECO**, **W_ZEN**, **W_AZEXPR**, **W_AZTRAIN**, **W_AZPROGR**: The weights used alongside each proxy. Make sure to align them with your list of proxies. They are float values that must add up to 1.0. They are set to zero as the default. <br />

Here is a usage example, based on the final setup adopted in the paper:

```
python ChimeraSearch.py --exp_name chimera-nas --population 100 --iterations 1000 --max_size 10 --features "macs" "meco" --w_macs 0.35 --w_meco 0.65
```

# Evaluating the final architectures from the paper

First of all, you need to open the "yaml" file for the corresponding dataset you want to test and modify the paths to the location you are using.
Then, you can run:
```
 python val.py --data ${DATASET}.yaml --model ${WEIGHTS}.pt --channels ${CH} --split ${SPLIT} --show_sequences ${SEQ} --img ${IMGH} ${IMGW}
```

**SPLIT**: val, test <br />
**DATASET**: You can find the "yaml" files inside the "data" folder. Example: "data/gen1/gen1_shist.yaml" <br />
**WEIGHTS**: They are stored in the "weights" folder. Example: "weights/gen1/chimera-n0-gen1.pt" <br />
**SEQ**: Number of sequences you want to see the predictions, default is -1 <br />
**CH**: Number of channels. In this work, the value 10 was adopted for all experiments <br />
**IMGH**, **IMGW**: Image size. We adopted IMGH=256, IMGW=320 <br />

# Datasets adopted in the work

The preprocessed versions from the GEN1 and PeDRo datasets adopted in this work can be found here: 

[preprocessed datasets](https://drive.google.com/file/d/1BwZU5eHsHk0yK0UPbwnz_huDHUF0uROY/view?usp=sharing)

# Single-GPU Training 

```
python train.py --batch ${BATCH} --nbs ${BATCH//2} --epochs ${NUM_EPOCH} --data ${DATASET}.yaml  --model ${MODEL_NAME}.yaml --channels ${CH} --name ${WANDB_RUN_NAME} --project ${WANDB_PROJECT_NAME}  --hyp ${HYP}.yaml --zoom_out ${Z} --flip ${F} --val_epoch ${VAL_EPOCH} --clip_length ${CLIP_LENGTH} --clip_stride ${CLIP_STRIDE} --img ${IMGH} ${IMGW}
```
where:

**BATCH**: Batch size <br />
**NUM_EPOCH**: Number of epochs <br />
**MODEL_NAME**: The "yaml" files are stored inside the folder "models". Example: "models/chimera-n0.yaml" <br />
**CH**: Number of channels. In all the experiments in this work, this number was set to 10. <br />
**HYP**: The "yaml" files for the hyperparameters are stored inside the folder "hyps". It contains some hyperparameters such as learning rate, weight decay, momentum, and the loss function multipliers. Example: "hyps/hyp_gen1.yaml" <br />
**F**: Horizontal flip probability <br />
**Z**: Zoom-out probability <br />
**VAL_EPOCH**: Number of epochs to perform validation <br />
**CLIP_LENGTH**: Length of the clips used for training <br />
**CLIP_STRIDE**: Distance between different clips. If equal to CLIP_LENGTH, clips will not present overlap. <br />
**IMGH**, **IMGW**: Image size. We adopted IMGH=256, IMGW=320 <br />

To accelerate the training, we adopted some tricks:  <br />
**1-** We validated only at each 10 epochs <br />
**2-** During training, instead of running the validation steps on full sequences, we divided the **val** set into batches that can be processed faster. <br />
**3-** On the training pipeline, only the final validation step over the **test** set is calculated over full sequences. <br />
**4-** Values reported in the paper that refer to the **val** set come from running **val.py** after training <br />

The factor **--nbs** stands for Normalized Batch Size. It is also present in the original Ultralytics repo and is utilized to make the training more robust to different batch sizes. Accordingly, 
the Weight Decay was set taking into consideration the **nbs** and the **clip length** according to:

```
W_Decay = W0*Batch_size*Clip_Length/NBS
```
Where **W0** is the weight decay defined in the hyperparameter files and **W_Decay** is the one adopted during training (and reported in the paper)

# Multi-GPU Training
```
torchrun --nnodes 1  --nproc_per_node ${GPUS}  train.py --device ${LIST_OF_GPUS} --batch ${BATCH} --nbs ${BATCH//2} --epochs ${NUM_EPOCH} --data ${DATASET}.yaml  --model ${MODEL_NAME}.yaml --channels ${CH} --name ${WANDB_RUN_NAME} --project ${WANDB_PROJECT_NAME}  --hyp ${HYP}.yaml --zoom_out ${Z} --flip ${F} --val_epoch ${VAL_EPOCH} --clip_length ${CLIP_LENGTH} --clip_stride ${CLIP_STRIDE}
```
where: <br />
**GPUS**: The number of GPUs in your node <br />
**LIST_OF_GPUS**: The list of the devices from your node. For example, for three GPUs, use [0,1,2] <br />


# Analysis of the Chimera Testbed

The reported results from the paper related to the Chimera Testbed are stored in the "Testbed" sub-folder. The file "testbed_results.csv" has all the proxies and training results from the testbed models. <br />
```
cd Testbed
```
The first step is to check the distribution of the mAPs across the different event representations: <br />
```
python plot_map_format.py
```
Then, you can check the different weights derived through Regression Trees in the paper: <br />
```
python testbed_regression.py
```
Finally, the Spearman and Kendall correlations, alongside the Mean-Squared Error over the 10% models from the testbed, are calculated through: <br /> 
```
python Testbed_Correlations.py
```

# Checking the NAS experiments

The folder "nas_experiments" contains more details about some NAS experiments, including tests with different weights. The experiments related to the final reported architectures are inside the sub-folder "macs035_meco065". Inside each folder, there are, for each parameter constraint adopted, the final population, the log with the scores for each of them, and the weights related to the top-5 ranked architectures from the ZS-NAS search after being trained with the PeDRo dataset. 

# Code Acknowledgements

- https://github.com/silvada95/ReYOLOv8

# Cite this work

To be uploaded soon

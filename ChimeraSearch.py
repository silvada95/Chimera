from EvolveUtils import *
import numpy as np
import yaml
import os
import json
import pandas as pd
import time
import random
import copy
import argparse
import glob 
import random
from sklearn.preprocessing import StandardScaler  # Import StandardScaler


ALLOWED_FEATURES = ["macs", "meco", "zen score", "az-train", "az-prog", "az-expr"]

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default="ablation_macs_only")
    parser.add_argument('--population', type=int, default=50)
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--max_size', type=int, default=10)
    parser.add_argument(
        "-f", "--features",
        nargs="+",                      # One or more values
        choices=ALLOWED_FEATURES,       # Restrict to allowed set
        metavar="FEATURE",
        help=(
            "One or more features to enable. "
            "Choices: " + ", ".join(ALLOWED_FEATURES) + ". "
            "If a feature contains a space, wrap it in quotes (e.g. \"zen score\")."
        ),
        required=True                   # Make it required (remove if optional)
    )
    
    parser.add_argument('--w_macs', type=float, default="0.35", help="weight multipliers for macs")
    parser.add_argument('--w_meco', type=float, default="0.65", help="weight multipliers for meco")
    parser.add_argument('--w_zen', type=float, default="0.0", help="weight multipliers for zen score")
    parser.add_argument('--w_azexpr', type=float, default="0.0", help="weight multipliers for az expr")
    parser.add_argument('--w_aztrain', type=float, default="0.0", help="weight multipliers for az train")
    parser.add_argument('--w_azprogr', type=float, default="0.0", help="weight multipliers for az progr")
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

args = parse_opt()


total = args.w_macs + args.w_meco + args.w_zen + args.w_azexpr + args.w_aztrain + args.w_azprogr
if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        parser.error(f"--w1 + --w2 + --w3 must sum to 1.0 (got {total:.12f})")


##################### CONFIG CONSTANTS #######################################


LIBRARY = {
"block": ["MambaVisionLayer","C2f",'MaxVitAttentionPairCl', "WaveMLPLayer"],
#"block": ["MambaVisionLayer","C2f", 'MaxVitAttentionPairCl'],
"Memory_cell": ["Conv_LSTM"],
"Channels" : [16,20,24,28,32,36,40,44,48],
"Repeats" : [1,2,3],
"Multiplier" : [1.0, 1.25, 1.33,1.50, 1.66, 1.75, 2.00],
"backbone_blocks" :  14,
"num_heads" : [1,2,3],
"head_multiplier" : [1.0, 1.25, 1.50,2.0]
}


#LIBRARY = {
#"block": ["MambaVisionLayer","C2f",'MaxVitAttentionPairCl', "WaveMLPLayer"],
#"Memory_cell": ["Conv_LSTM"],
#"Channels" : [12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48],
#"Repeats" : [1,2,3],
#"Multiplier" : [1.0, 1.10, 1.15, 1.20 ,1.25, 1.30,  1.35, 1.40, 1.45,1.50, 1.55, 1.60, 1.65, 1.70, 1.75, 1.80, 1.85, 1.90, 1.95, 2.00],
#"backbone_blocks" :  14,
#"num_heads" : [1,2,3],
#"head_multiplier" : [1.0, 1.25, 1.50,2.0]
#}


POPULATION_SIZE = args.population
ITERATIONS = args.iterations
MAX_SIZE = args.max_size*1000000
features = args.features

HEADER = ['nc: 80', 
        'depth_multiple: 1.0', 
        'width_multiple: 1.0', 
        'act: nn.SiLU()']

CH_IN = 5

DEVICE = "cuda:0"
################### Initialize the Population #################################

individuals = []
macs = []
tags = []
params = []

path = []
mlps = []
mambas = []
maxvits = []
zen_score = []
az_progressivity = []
az_expressivity = []
az_trainability = []
convs = []
meco = []
scaler = StandardScaler()

i = 0


def makename(exp_name):
  
     paths = sorted(glob.glob(os.path.join(os.getcwd(), exp_name, "*/")))

     if not paths:
       os.makedirs(os.path.join(os.getcwd(), exp_name, "exp000001"))
       return os.path.join(os.getcwd(), exp_name, "exp000001")
     else:
       if len(paths) == 1:
          os.makedirs(os.path.join(os.getcwd(), exp_name, "exp000002"))
          return os.path.join(os.getcwd(), exp_name, "exp000002")
       else:
        ref_string = os.path.join(os.getcwd(), exp_name, "exp")

        idx = int(paths[-1][len(ref_string):len(paths[-1]) - 1])
        
        os.makedirs(os.path.join(os.getcwd(), exp_name, "exp" + str(idx+1).zfill(6)))
        return os.path.join(os.getcwd(), exp_name, "exp" + str(idx+1).zfill(6))
   

exp_name = makename(args.exp_name)

t0 = time.time()

#features = ["meco", "macs"]

meco_mult = args.w_meco
macs_mult = args.w_macs
zen_mult = args.w_zen
aztrain_mult = args.w_aztrain
azprogr_mult = args.w_azprogr
azexpr_mult = args.w_azexpr
weights = {"meco": meco_mult, "macs":macs_mult, "zen score": zen_mult, "az-train": aztrain_mult, "az-prog": azprogr_mult, "az-expr": azexpr_mult}

compute_meco = "meco" in features
compute_az = any(az_key in features for az_key in ["az-train", "az-prog", "az-expr"])
compute_zen = "zen score" in features


    
while i < POPULATION_SIZE:
    print(" current iteration ", i)
    individual = GenerateIndividual(LIBRARY)
    tag = "ind_" + str(i)
    cfg = os.path.join(os.getcwd(), exp_name, tag + ".yaml")
    individual2yaml(individual, HEADER, cfg, MAX_SIZE)
    
    try:
     if not CheckIndividualSize(cfg, CH_IN, DEVICE, MAX_SIZE):
            os.remove(cfg)
     else:
            # Call ProfileIndividual with options you want
            results = ProfileIndividual(cfg, CH_IN, compute_meco=compute_meco, compute_zen=compute_zen, compute_az=compute_az)

            individuals.append(individual)
            vec = individual2vec(individual)
            convs.append(vec[0])
            mlps.append(vec[1])
            mambas.append(vec[2])
            maxvits.append(vec[3])
            tags.append(tag)
            path.append(cfg)

            # Append macs and params, conditionally check
            macs_value = results.get('macs', None)
            params_value = results.get('params', None)
            macs.append(macs_value)
            params.append(params_value)

            # Append optional values conditionally
            if 'meco' in results:
                meco.append(results['meco'])
            if 'zen' in results:
                zen_score.append(results['zen'])
            if 'az_progressivity' in results:

                az_progressivity.append(results['az_progressivity'])

            if 'az_expressivity' in results:
                az_expressivity.append(results['az_expressivity'])
            if 'az_trainability' in results:
                az_trainability.append(results['az_trainability'])

            i += 1
    except Exception as e:
        print(f"Error occurred: {e}")
        os.remove(cfg)  
        pass 

print(" time elapsed to generate initial population :", time.time() - t0)

# Create the final data dictionary, ensuring to remove unused keys
data = {
    "tag": tags,
    "path": path,
    "individual": individuals,
    "macs": macs,
    "params": params,
    "meco": meco if 'meco' in results else None,
    "zen score": zen_score if 'zen score' in features else None,  # Only include if set
    "az-prog": az_progressivity if 'az-prog' in features else None,
    "az-expr": az_expressivity if 'az-expr' in features else None,
    "az-train": az_trainability if 'az-train' in features else None,
    "mlps": mlps,
    "mambas": mambas,
    "maxvits": maxvits,
    "convs": convs
}

pd_ = pd.DataFrame(data=data)


it = 0 


aligned_weights = np.array([float(weights[f]) for f in features])
X = pd_[features].to_numpy()
X_norm = scaler.fit_transform(X)
cscore = aligned_weights @ X_norm.T

pd_.insert(6, "cscore",cscore)


t0 = time.time()


while it < ITERATIONS:
    
    
    #### GENERATE A NEW INDIVIDUAL THROUGH MUTATION  
    idx = random.randint(0, POPULATION_SIZE - 1)
    tag = "ind_" + str(POPULATION_SIZE + it)
    new_ind = copy.deepcopy(pd_["individual"][idx])

    new_ind = Mutate(new_ind, LIBRARY).apply()


    cfg = os.path.join(os.getcwd(), exp_name, str(tag) + ".yaml")
    individual2yaml(new_ind, HEADER, cfg, MAX_SIZE)
    try:
     if not CheckIndividualSize(cfg, CH_IN, DEVICE, MAX_SIZE):
        os.remove(cfg)
     else:
        # Specify which metrics to compute based on your needs
        results = ProfileIndividual(cfg, CH_IN, compute_meco=compute_meco, compute_zen=compute_zen, compute_az=compute_az)
        
        vec = individual2vec(new_ind)
        # Append optional values conditionally
        if 'meco' in results:
                meco = results['meco']
        if 'zen' in results:
                zen_score = results['zen']
        if 'az_progressivity' in results:
               az_progressivity = results['az_progressivity']
        if 'az_expressivity' in results:
                az_expressivity = results['az_expressivity']
        if 'az_trainability' in results:
                az_trainability = results['az_trainability']

        # Initialize the new row with optional keys
        new_row = {
            "tag": tag,
            "path": cfg,
            "individual": new_ind,
            "macs": results.get('macs', None),
            "params": results.get('params', None),
            "meco": results.get('meco', None),  # Will be None if not calculated
            "zen score": zen_score if 'zen score' in features else None,  # Only include if set
            "az-prog": az_progressivity if 'az-prog' in features else None,
            "az-expr": az_expressivity if 'az-expr' in features else None,
            "az-train": az_trainability if 'az-train' in features else None,
            "mlps": vec[1],
            "mambas": vec[2],
            "maxvits": vec[3],
            "convs": vec[0],
            "cscore": 0,  # Initialized to zero as specified
            
        }


        # Check if the individual is different and append the new row
        if new_ind != pd_["individual"][idx]:
            pd_ = pd_._append(new_row, ignore_index=True)
            
            X = pd_[features].to_numpy()
            X_norm = scaler.fit_transform(X)
            cscore = aligned_weights @ X_norm.T
            pd_["cscore"] = cscore

            # Sort by cscore and remove the lowest

            pd_ = pd_.sort_values(by=['cscore'], ascending=False)

            # Remove the last element based on sorted cscore
            os.remove(list(pd_["path"][pd_.tail(1).index])[0])
            pd_ = pd_.iloc[:-1]
            print(pd_)
            pd_ = pd_.reset_index(drop=True)
            
            it += 1
    except Exception as e:
     print(f"Error occurred: {e}")
     os.remove(cfg)
     pass
     

pd_.to_csv(os.path.join(exp_name, "individuals_it_" + str(ITERATIONS) + "_ind_" + str(POPULATION_SIZE) + "size_" + str(args.max_size) + "_macs_" + str(macs_mult)[2:6] +  "_meco_" + str(meco_mult)[2:6]  +".csv"))
print("time necessary to perform the search for 1000 iterations", time.time() - t0)  






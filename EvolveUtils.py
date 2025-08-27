import torch
import torch.nn as nn
import math
import numpy as np
import random
import yaml
import os
from nas_utils.ModelDebug import NAS_MEASURE_MODEL
from ultralytics.nn.tasks import DetectionModel2
from thop import profile as profile
import pandas as pd
from nas_utils.scores.compute_zen_score import compute_nas_score
import json
import glob
from nas_utils.scores.compute_meco_score import compute_meco_score
from nas_utils.scores.compute_general_aznas_score import compute_aznas_score
'''
BASIC TEMPLATE - SINGLE-PATHWAY MODEL, 

Structure based on ReYOLOv8, which adopts 
the YOLOv5 ð??? by Ultralytic template GPL-3.0 license

# Parameters
nc: 80  # number of classes
depth_multiple: 0.5  # scales module repeats
width_multiple: 0.25  # scales convolution channels
act: nn.SiLU()

backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]]  # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]  # 1-P2/4
  - [-1, 3, C2f, [128, True]]
  - [-1, 1, Conv_LSTM, [128]]
  - [-1, 1, Conv, [256, 3, 2]]  # 4-P3/8
  - [-1, 6, C2f, [256, True]]
  - [-1, 1, Conv_LSTM, [256]]
  - [-1, 1, Conv, [512, 3, 2]]  # 7-P4/16
  - [-1, 6, MaxVitAttentionPairCl, [512]]
  - [-1, 1, Conv_LSTM, [512]]
  - [-1, 1, Conv, [1024, 3, 2]]  # 10-P5/32
  - [-1, 3, C2f, [1024, True]]
  - [-1, 1, Conv_LSTM, [1024]]
  - [-1, 1, SPPF, [1024, 5]]  # 13


head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 9], 1, Concat, [1]]  # cat backbone P4
  - [-1, 3, C2f, [512]]  # 16

  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 6], 1, Concat, [1]]  # cat backbone P3
  - [-1, 3, C2f, [256]]  # 19 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 16], 1, Concat, [1]]  # cat head P4
  - [-1, 3, C2f, [512]]  # 22 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 13], 1, Concat, [1]]  # cat head P5
  - [-1, 3, C2f, [1024]]  # 25 (P5/32-large)

  - [[19, 22, 25], 1, Detect, [nc]]  # Detect(P3, P4, P5)


I assembles positions from 0 to 12... From a total of 27

The structure from the Head does not change... However, the number of channels can be set up
'''

def make_divisible(x, divisor):
  
    return math.ceil(x / divisor) * divisor

def GenerateIndividual(library):
 
 
 
 
 block = library["block"]
 Memory_cell = library["Memory_cell"]
 Channels = library["Channels"]
 Repeats = library["Repeats"]
 Multiplier = library["Multiplier"]
 backbone_blocks = library["backbone_blocks"]
 num_heads = library["num_heads"]
 head_multiplier = library["head_multiplier"]
  
 individual = {}
 hd = -1
 for i in range(backbone_blocks):
    
    if i == 0:
       ch0 = random.choice(Channels)
       ch0 = make_divisible(ch0,2)
       individual["0"] = ch0
       
    if i == 1:
       mult = random.choice(Multiplier)
       ch = make_divisible(int(mult*ch0),2)
       individual["1"] = ch
       
    if i == 2: 
       blk = random.choice(block)
       layer = random.choice(Repeats)
       if blk == 'C2f': 
         individual["2"] = {"block": blk, "repeats": layer, "ch": ch}
       elif  blk == 'WaveMLPLayer':
         
         individual["2"] = {"block": blk, "repeats": layer, "ch": ch}
       elif blk == "MambaVisionLayer":
         hd = random.choice(num_heads)
         individual["2"] = {"block": blk, "repeats": layer, "ch": ch, "hd": hd}
       else:
         ch = make_divisible(ch, 8)
         individual["1"] = ch
         individual["2"] = {"block": blk, "ch": ch}
    if i == 3: 
       blk = random.choice(Memory_cell)
       individual["3"] = {"block": blk, "ch": ch}
    if i == 4:
       mult = random.choice(Multiplier)
       ch = make_divisible(int(mult*ch),2)
       individual["4"] = ch
    if i == 5: 
       blk = random.choice(block)
       layer = random.choice(Repeats)
       if blk == 'C2f': 

         individual["5"] = {"block": blk, "repeats": layer, "ch": ch}
       elif  blk == 'WaveMLPLayer':

         individual["5"] = {"block": blk, "repeats": layer, "ch": ch}
       elif blk == "MambaVisionLayer":
         if hd > 0 :
          hd = int(hd*random.choice(head_multiplier))
         else: 
          hd = random.choice(num_heads)
          hd = int(hd*random.choice(head_multiplier))
          
         individual["5"] = {"block": blk, "repeats": layer, "ch": ch, "hd": hd}

       else:
         ch = make_divisible(ch, 8)
         individual["4"] = ch
         individual["5"] = {"block": blk, "ch": ch}
    if i == 6: 
       blk = random.choice(Memory_cell)
       individual["6"] = {"block": blk, "ch": ch}
       
    if i == 7:
       mult = random.choice(Multiplier)
       ch = make_divisible(int(mult*ch),2)
       individual["7"] = ch
    if i == 8: 
       blk = random.choice(block)
       layer = random.choice(Repeats)
       if blk == 'C2f': 
         individual["8"] = {"block": blk, "repeats": layer, "ch": ch}
       elif  blk == 'WaveMLPLayer':
         individual["8"] = {"block": blk, "repeats": layer, "ch": ch}
       elif blk == "MambaVisionLayer":
         if hd > 0 :
          hd = int(hd*random.choice(head_multiplier))
         else: 
          hd = random.choice(num_heads)
          hd = int(hd*random.choice(head_multiplier))
         individual["8"] = {"block": blk, "repeats": layer, "ch": ch, "hd": hd} 
       else:
         ch = make_divisible(ch, 8)
         individual["7"] = ch
         individual["8"] = {"block": blk, "ch": ch}
    if i == 9: 
       blk = random.choice(Memory_cell)

       individual["9"] = {"block": blk, "ch": ch}
    if i == 10:
       mult = random.choice(Multiplier)
       ch = make_divisible(int(mult*ch),2)

       individual["10"] = ch
    if i == 11: 
       blk = random.choice(block)
       layer = random.choice(Repeats)
       if blk == 'C2f': 

         individual["11"] = {"block": blk, "repeats": layer, "ch": ch}
       elif  blk == 'WaveMLPLayer':

         individual["11"] = {"block": blk, "repeats": layer, "ch": ch}
       elif blk == "MambaVisionLayer":
         if hd > 0 :
          hd = int(hd*random.choice(head_multiplier))
         else: 
          hd = random.choice(num_heads)
          hd = int(hd*random.choice(head_multiplier))
         individual["11"] = {"block": blk, "repeats": layer, "ch": ch, "hd": hd}  

       else:
         ch = make_divisible(ch, 8)
         individual["10"]
         individual["11"] = {"block": blk, "ch": ch}
    if i == 12: 
       blk = random.choice(Memory_cell)

       individual["12"] = {"block": blk, "ch": ch}
    if i == 13: 
       individual["13"] = {"ch": ch}
  

 return individual 

def individual2vec(individual):
    ##### 0 -> C2f, 1 -> MLP, 2 -> Mamba, 3-> MaxViT
    block_vec = np.zeros((4,1))
    for ind in individual:

        try:
            keys = individual[str(ind)].keys()
            
            if "block" in keys:

                if individual[str(ind)]["block"] == "C2f":
                   block_vec[0,0] += 1
                elif individual[str(ind)]["block"] == "WaveMLPLayer":
                   block_vec[1,0] += 1
                elif individual[str(ind)]["block"] == "MambaVisionLayer":
                   block_vec[2,0] += 1
                elif  individual[str(ind)]["block"] == 'MaxVitAttentionPairCl':
                   block_vec[3,0] += 1
        except:
               pass
               
    
    return block_vec    
      


def individual2yaml(individual, data,destination, max_size):

 ######################################### BACKBONE #############################################################

 backbone_ls = []
 hd = -1
 backbone_blocks = len(individual.keys()) 
 for i in range(backbone_blocks):

    if i == 0:
       ch0 = individual["0"]
       backbone_ls.append(' - [-1,1, Conv, ['+str(ch0)+',3,2]] #0-P1/2')

    if i == 1:

       ch = individual["1"]
       backbone_ls.append(' - [-1,1, Conv, ['+str(ch)+',3,2]] #1-P2/4')
 
    if i == 2: 
       blk = individual["2"]["block"]
       
       ch = individual["2"]["ch"]
       if blk == 'C2f': 
         layer = individual["2"]["repeats"]
         backbone_ls.append(' - [-1,'+str(layer)+', '+blk+', ['+str(ch)+',True]]')
         

       elif  blk == 'WaveMLPLayer':
         layer = individual["2"]["repeats"]
         backbone_ls.append(' - [-1,'+str(layer)+', '+blk+', ['+str(ch)+']]')
         

       elif blk == "MambaVisionLayer":
         hd = individual["2"]["hd"]
         layer = individual["2"]["repeats"]
         backbone_ls.append(' - [-1,'+str(1)+', '+blk+', ['+str(ch)+','+str(layer)+', '+str(hd)+','+str(8)+']]')
         
       else:
         backbone_ls.append(' - [-1,'+str(1)+', '+blk+', ['+str(ch)+']]')
         
         
    if i == 3: 
       blk = individual["3"]["block"]
       backbone_ls.append(' - [-1,1, '+blk+', ['+str(ch)+']]')
       
    if i == 4:

       ch = individual["4"]
       backbone_ls.append(' - [-1,1, Conv, ['+str(ch)+',3,2]] #4-P3/8')
       
    if i == 5: 
       blk = individual["5"]["block"]
       
       ch = individual["5"]["ch"]
       if blk == 'C2f': 
         layer = individual["5"]["repeats"]
         backbone_ls.append(' - [-1,'+str(layer)+', '+blk+', ['+str(ch)+',True]]')

       elif  blk == 'WaveMLPLayer':
         layer = individual["5"]["repeats"]
         backbone_ls.append(' - [-1,'+str(layer)+', '+blk+', ['+str(ch)+']]')

       elif blk == "MambaVisionLayer":
         layer = individual["5"]["repeats"]
         hd = individual["5"]["hd"]
         backbone_ls.append(' - [-1,'+str(1)+', '+blk+', ['+str(ch)+','+str(layer)+', '+str(hd)+','+str(8)+']]')
         
       else:
         backbone_ls.append(' - [-1,'+str(1)+', '+blk+', ['+str(ch)+']]')

    if i == 6: 
       blk = individual["6"]["block"]
       backbone_ls.append(' - [-1,1, '+blk+', ['+str(ch)+']]')

       
    if i == 7:
       ch = individual["7"]
       backbone_ls.append(' - [-1,1, Conv, ['+str(ch)+',3,2]] #7-P4/16')

    if i == 8: 
       blk = individual["8"]["block"]

       ch = individual["8"]["ch"]
       if blk == 'C2f': 
         layer = individual["8"]["repeats"]
         backbone_ls.append(' - [-1,'+str(layer)+', '+blk+', ['+str(ch)+',True]]')

       elif  blk == 'WaveMLPLayer':
         layer = individual["8"]["repeats"]
         backbone_ls.append(' - [-1,'+str(layer)+', '+blk+', ['+str(ch)+']]')

       elif blk == "MambaVisionLayer":
         layer = individual["8"]["repeats"]
         hd = individual["8"]["hd"]
         backbone_ls.append(' - [-1,'+str(1)+', '+blk+', ['+str(ch)+','+str(layer)+', '+str(hd)+','+str(8)+']]')
       else:
         backbone_ls.append(' - [-1,'+str(1)+', '+blk+', ['+str(ch)+']]')

    if i == 9: 
       blk = individual["9"]["block"]
       backbone_ls.append(' - [-1,1, '+blk+', ['+str(ch)+']]')

    if i == 10:

       ch = individual["10"]
       backbone_ls.append(' - [-1,1, Conv, ['+str(ch)+',3,2]] #10-P5/16')

    if i == 11: 
       blk = individual["11"]["block"]
       
       ch = individual["11"]["ch"]
       if blk == 'C2f': 
         layer = individual["11"]["repeats"]
         backbone_ls.append(' - [-1,'+str(layer)+', '+blk+', ['+str(ch)+',True]]')

       elif  blk == 'WaveMLPLayer':
         layer = individual["11"]["repeats"]
         backbone_ls.append(' - [-1,'+str(layer)+', '+blk+', ['+str(ch)+']]')

       elif blk == "MambaVisionLayer":
         layer = individual["11"]["repeats"]
         ch = individual["11"]["hd"]
         backbone_ls.append(' - [-1,'+str(1)+', '+blk+', ['+str(ch)+','+str(layer)+', '+str(hd)+','+str(8)+']]')
       else:

         backbone_ls.append(' - [-1,'+str(1)+', '+blk+', ['+str(ch)+']]')
    if i == 12: 
       blk = individual["12"]["block"]
       backbone_ls.append(' - [-1,1, '+blk+', ['+str(ch)+']]')

    if i == 13: 
       
       backbone_ls.append(' - [-1,1, SPPF, ['+str(ch)+',5]]')
       
 



 ######################################### HEAD #############################################################


 head_ls = []
 

 if max_size > 14:
    head_m = [8,4,12]
 else:
    head_m = [8,4,16]
  
 for i in range(14,27):
    
    if i == 14: 
       head_ls.append(' - [-1, 1, nn.Upsample, [None, 2, nearest]]')
    if i == 15:
       head_ls.append(' - [[-1, 9], 1, Concat, [1]]  # cat backbone P4')
    if i == 16: 
       head_ls.append(' - [-1,1,C2f, ['+str(head_m[0]*ch0)+']]')
    if i == 17: 
       head_ls.append(' - [-1, 1, nn.Upsample, [None, 2, nearest]]')
    if i == 18:
       head_ls.append(' - [[-1, 6], 1, Concat, [1]]  # cat backbone P3')
    if i == 19: 
       head_ls.append(' - [-1,1,C2f, ['+str(head_m[1]*ch0)+']] # P3/8-small')
    if i == 20: 
       head_ls.append(' - [-1, 1, Conv, ['+str(head_m[1]*ch0)+', 3, 2]]')
    if i == 21:
       head_ls.append(' - [[-1, 16], 1, Concat, [1]] # cat head P4')
    if i == 22: 
       head_ls.append(' - [-1,1,C2f, ['+str(head_m[0]*ch0)+']] # P4/16-Medium')
    if i == 23: 
       head_ls.append(' - [-1, 1, Conv, ['+str(head_m[0]*ch0)+', 3, 2]]')
    if i == 24:
       head_ls.append(' - [[-1, 13], 1, Concat, [1]] # cat head P5')
    if i == 25: 
       head_ls.append(' - [-1,1,C2f, ['+str(head_m[2]*ch0)+']] # P5/32-Large')
    if i == 26:
       head_ls.append(' - [[19, 22, 25], 1, Detect, [nc]]  # Detect(P3, P4, P5)')

 with open(destination, 'w') as yaml_file:

     for item in data:
        yaml_file.write(f"{item}\n")
        
     yaml_file.write(f"backbone:\n")
     for item in backbone_ls:
         yaml_file.write(f"{item}\n")
         
     yaml_file.write(f"head:\n")
     for item in head_ls:
         yaml_file.write(f"{item}\n")
 


#individual =  GenerateIndividual(library)



class Mutate():

      def __init__(self, Individual, library):
          
          self.Individual = Individual
          self.block = library["block"]
          self.Memory_cell = library["Memory_cell"]
          self.Channels = library["Channels"]
          self.Repeats = library["Repeats"]
          self.Multiplier = library["Multiplier"]
          self.num_heads = library["num_heads"]
          self.head_multiplier = library["head_multiplier"]
      
          self.sort_interval = [0,1,2,4,5,7,8,10,11]
      def modify_ch_only(self, key):
      
            new_ch = random.choice(self.Channels)
            old_ch = self.Individual[str(0)] 
            self.Individual[str(0)] = new_ch
            
            self.Individual[str(1)] = int((self.Individual[str(1)] // old_ch)*new_ch)
            self.Individual[str(2)]["ch"] = int((self.Individual[str(1)] // old_ch)*new_ch)     
            self.Individual[str(3)]["ch"] = int((self.Individual[str(1)] // old_ch)*new_ch)    
            
            self.Individual[str(4)] = int((self.Individual[str(4)] // old_ch)*new_ch)
            self.Individual[str(5)]["ch"] = int((self.Individual[str(4)] // old_ch)*new_ch)     
            self.Individual[str(6)]["ch"] = int((self.Individual[str(4)] // old_ch)*new_ch)   
            
            self.Individual[str(7)] = int((self.Individual[str(7)] // old_ch)*new_ch)
            self.Individual[str(8)]["ch"] = int((self.Individual[str(7)] // old_ch)*new_ch)     
            self.Individual[str(9)]["ch"] = int((self.Individual[str(7)] // old_ch)*new_ch)   
                  
            self.Individual[str(10)] = int((self.Individual[str(10)] // old_ch)*new_ch)
            self.Individual[str(11)]["ch"] = int((self.Individual[str(10)] // old_ch)*new_ch)     
            self.Individual[str(12)]["ch"] = int((self.Individual[str(10)] // old_ch)*new_ch)          
            self.Individual[str(13)]["ch"] = int((self.Individual[str(10)] // old_ch)*new_ch)

      def modify_mult_only(self, key):

               find_ = False
               while not find_:
                  if key != 10:
                    mult_ = random.choice(self.Multiplier)
                    
                    if key == 1:
                     new_ch = int(mult_*self.Individual[str(key-1)])
                     print(new_ch, self.Individual[str(key - 1)], self.Individual[str(key +3)])
                     if (new_ch >= self.Individual[str(key - 1)]) and (new_ch <= self.Individual[str(key + 3)]): 
                        self.Individual[str(key)] = new_ch
                        self.Individual[str(key + 1)]["ch"] = new_ch
                        self.Individual[str(key + 2)]["ch"] = new_ch
                        find_ = True
                        
                    else:
                     new_ch = int(mult_*self.Individual[str(key-1)]["ch"])
                     print(new_ch, self.Individual[str(key - 1)]["ch"], self.Individual[str(key +3)])
                     if (new_ch >= self.Individual[str(key - 1)]["ch"]) and (new_ch <= self.Individual[str(key + 3)]): 
                        self.Individual[str(key)] = new_ch
                        self.Individual[str(key + 1)]["ch"] = new_ch
                        self.Individual[str(key + 2)]["ch"] = new_ch
                        find_ = True 
                        
                  else: 
                    mult_ = random.choice(self.Multiplier)
                    new_ch = int(mult_*self.Individual[str(key-1)]["ch"])
                    if (new_ch >= self.Individual[str(key - 1)]["ch"]): 
                        self.Individual[str(key)] = new_ch
                        self.Individual[str(key + 1)]["ch"] = new_ch
                        self.Individual[str(key + 2)]["ch"] = new_ch
                        self.Individual[str(key + 3)]["ch"] = new_ch
                        find_ = True
             
      def modify_processing(self, key):
          try:
           old_keys = self.Individual[str(key)]["block"].keys()
          except:
             old_keys = []
             pass
          self.Individual[str(key)]["block"] = random.choice(self.block)
          
          
          if self.Individual[str(key)]["block"] == "C2f":
             self.Individual[str(key)]["repeats"] = random.choice(self.Repeats)
             
             if "hd" in old_keys:
                 del self.Individual[str(key)]["hd"]
                 
          elif  self.Individual[str(key)]["block"] == "WaveMLPLayer":
             self.Individual[str(key)]["repeats"] = random.choice(self.Repeats)      
             if "hd" in old_keys:
                 del self.Individual[str(key)]["hd"]
                 
          elif self.Individual[str(key)]["block"] == "MambaVisionLayer":
               self.Individual[str(key)]["repeats"] = random.choice(self.Repeats) 
               self.Individual[str(key)]["hd"] = random.choice(self.num_heads)
               
          else:     
               if "hd" in old_keys:
                 del self.Individual[str(key)]["hd"]
               if "repeats" in old_keys:
                 del self.Individual[str(key)]["repeats"]  
                 
      def apply(self):
           key = random.choice(self.sort_interval)

           if key == 0: 
              self.modify_ch_only(key)
           elif key in [1,4,7,10]:
              self.modify_mult_only(key)
           else:
              self.modify_processing(key)
              
           return self.Individual                
        

def CheckIndividualSize(cfg, ch, device, max_size):

     #the_model = NAS_MEASURE_MODEL(cfg, ch, device)
     the_model = DetectionModel2(cfg, imgsz = [256,320],ch=ch, nc=1, verbose=False).to(device)
     in_ = torch.rand(1,ch,256,320).to(device)
     macs_, params_ = profile(the_model, (in_, {"0":torch.empty(0), "1":torch.empty(0), "2":torch.empty(0), "3": torch.empty(0)}), verbose=False)
     
     
     return  params_ < max_size

def ProfileIndividual(cfg, ch, compute_macs=True, compute_params=True, compute_meco=False, 
                      compute_zen=False, compute_az=False):
    the_model = NAS_MEASURE_MODEL(cfg, ch, [256,320], "cuda:0")
    
    results = {}

    if compute_macs:
        in_ = torch.rand(1, ch, 256, 320).to("cuda:0")
        macs_, params_ = profile(
            the_model, 
            (in_, {"0": torch.empty(0), "1": torch.empty(0), "2": torch.empty(0), "3": torch.empty(0)}), 
            verbose=False
        )
        results['macs'] = macs_

    if compute_params:
        results['params'] = params_

    if compute_meco:
        meco = compute_meco_score(model=cfg, ch=ch, batch=1, imgsz=[256, 320], device=0)
        results['meco'] = meco

    if compute_zen:
        repeat_times = 1
        batch_size = 32
        gpu = 0
        
        zen = compute_nas_score(gpu=gpu, ch=ch, model=the_model, mixup_gamma=1e-2,
                                 resolution=[256, 320], batch_size=batch_size, repeat=repeat_times, fp16=False)['avg_nas_score']
        results['zen'] = zen

    if compute_az:
        az = compute_aznas_score(model=the_model, gpu=0, event_data=[], files=[], encoding=[], ch=ch, actual_res=[346,260], target_res=[256,320], batch_size=4, init_method = 'kaiming_norm_fanin', fp16=False)
        results['az_progressivity'] = az["progressivity"]
        results['az_expressivity'] = az["expressivity"]
        results['az_trainability'] = az["trainability"]
    

    return results






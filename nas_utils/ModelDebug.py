import torch
import torch.nn as nn

import numpy as np
import random
import yaml

from ultralytics.nn.tasks import DetectionModel2
from thop import profile as prof

from ultralytics.nn.modules import *

class NAS_MEASURE_MODEL(nn.Module):
      
      def __init__(self,cfg,ch,img=[256,320],device="cuda:0"):
          super().__init__() 

          self.model = DetectionModel2(cfg, imgsz = img,ch=ch, nc=1, verbose=False).model[:14].to("cuda:0")
          
      def forward(self, x, hidden_states):
          self.lstm_layer = 0 

          for i in range(len(self.model)):

              if self.model[i].type == "ultralytics.nn.modules.Conv_LSTM":


                 
                 hidden_states[str(self.lstm_layer)] = self.model[i](x,hidden_states[str(self.lstm_layer)])

                 x = hidden_states[str(self.lstm_layer)][0]
                 self.lstm_layer += 1
              elif self.model[i].type == "ultralytics.nn.ssm.s5_model.S5BlockWrapper":


                 x, hidden_states[str(self.lstm_layer)] = self.model[i](x,hidden_states[str(self.lstm_layer)])

                 self.lstm_layer += 1
              else:

                 x = self.model[i](x)

          
          return x, hidden_states
      
      
      def extract_cell_features(self, inputs):
          self.lstm_layer = 0 
          x = inputs[0]
          hidden_states = inputs[1]
          cell_features = []

          for i in range(len(self.model)):

              if self.model[i].type == "ultralytics.nn.modules.Conv_LSTM":


                 
                 hidden_states[str(self.lstm_layer)] = self.model[i](x,hidden_states[str(self.lstm_layer)])

                 x = hidden_states[str(self.lstm_layer)][0]
                 
                 self.lstm_layer += 1
              elif self.model[i].type == "ultralytics.nn.ssm.s5_model.S5BlockWrapper":


                 x, hidden_states[str(self.lstm_layer)] = self.model[i](x,hidden_states[str(self.lstm_layer)])

                 self.lstm_layer += 1
              else:
                 x = self.model[i](x)
              cell_features.append(x)
          
          return cell_features
      

# def __init__(self,
#                 ch_in,
#                 dim,
#                 depth,
#                 num_heads,
#                 window_size,

#model = NAS_MEASURE_MODEL(cfg, ch=5, device="cuda:0")
#model = MambaVisionLayer(10,10,3,1,8).to("cuda:0")
#print(model)
#x = torch.rand(2,5,256,320).to("cuda:0")
#y = model.extract_cell_features([x,hidden_states])
#print(len(y))
#y = model(x)
#y = model(x, {"0": torch.empty(0), "1": torch.empty(0), "2": torch.empty(0), "3": torch.empty(0)})

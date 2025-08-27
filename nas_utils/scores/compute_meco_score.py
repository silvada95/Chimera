
import numpy as np
import os
import sys
import torch
import torch.nn.functional as f
from operator import mul
from functools import reduce
import copy
from ModelDebug import NAS_MEASURE_MODEL




class Jocab_Scorer:
    def __init__(self, gpu):
        self.gpu = gpu
        self.hidden_states = {"0": torch.empty(0), "1": torch.empty(0), "2": torch.empty(0), "3": torch.empty(0)}
        print('Jacob score init')

    def score(self, model, input):
        batch_size = input.shape[0]
        model.K = torch.zeros(batch_size, batch_size).cuda()

        input = input.cuda()
        with torch.no_grad():
            model(input, self.hidden_states)
        score = self.hooklogdet(model.K.cpu().numpy())

        #print(score)
        return score

    def setup_hooks(self, model, batch_size):
        #initalize score 
        model = model.to(torch.device('cuda',self.gpu))
        model.eval()
        model.K = torch.zeros(batch_size, batch_size).cuda()
        def counting_forward_hook(module, inp, out):
            try:
                # if not module.visited_backwards:
                #     return
             if isinstance(inp, tuple):
                inp = inp[0]
             inp = inp.view(inp.size(0), -1)
             x = (inp > 0).float()
             K = x @ x.t()
             K2 = (1.-x) @ (1.-x.t())
             model.K = model.K + K + K2
            except:
                   pass

        for name, module in model.named_modules():

            if "torch.nn.modules.activation.SiLU" in str(type(module)):

                module.register_forward_hook(counting_forward_hook)
                #module.register_backward_hook(counting_backward_hook)

    def hooklogdet(self, K, labels=None):
        s, ld = np.linalg.slogdet(K)
        return ld
        
       
def compute_meco_score(model, ch, batch, imgsz, device):
 scorer = Jocab_Scorer(device)


 model = NAS_MEASURE_MODEL(model, ch=ch, device="cuda:"+str(device))
 
 score = scorer.setup_hooks(model, batch)
 score = scorer.score(model, torch.rand(batch,ch,imgsz[0],imgsz[1]))
 return score


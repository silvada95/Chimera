import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import torch
from torch import nn
import numpy as np
from format_utils import taf_cuda, vtei, mdes, shist
import math
def create_event_input(event_file, target_res, actual_res, ch, encoding, device, dtype):    
    
            events = np.load(event_file)
            events = np.array([(events[i,0],events[i,1],events[i,2],events[i,3]) for i in range(len(events))], dtype = [('t','f8'), ('x','i4'), ('y','i4'), ('p','i1')])
            height = actual_res[0]
            width = actual_res[1]
                        
            if encoding == "mdes":
                 input = mdes(events, height, width, ch, device)
            elif encoding == "shist":
                 input = shist(events, height, width, ch, device)
            elif encoding == "vtei":
                 input = vtei(events, height, width, ch, device)
            elif encoding == "taf":
                  input = taf_cuda(events, height, width, ch, device)
            else:
                  raise ValueError("Not Implemented")      
            input = input.to(dtype=dtype)

            input = (torch.nn.functional.interpolate(input.view(input.shape[0],1,actual_res[0], actual_res[1]),scale_factor=(float(target_res[0]/actual_res[0]),float(target_res[1]/actual_res[1])), mode='area')) 
            return ((input.view(input.shape[0],input.shape[2],input.shape[3])))

def mdes(events, height, width, ch, device):
          x = torch.from_numpy(np.clip(events["x"].astype(np.int16).copy(),0,width-1))
          y = torch.from_numpy(np.clip(events["y"].astype(np.int16).copy(),0,height-1))
          t = torch.from_numpy(events["t"].astype(np.int64))
          p  = torch.from_numpy(events["p"].astype(np.int8))
            
          return vis.mdes(x, y, t, p, ch, height, width).to(device)

def shist(events, height, width, ch, device):
          x = torch.from_numpy(np.clip(events["x"].astype(np.int16).copy(),0,width-1))
          y = torch.from_numpy(np.clip(events["y"].astype(np.int16).copy(),0,height-1))
          t = torch.from_numpy(events["t"].astype(np.int64))
          p  = torch.from_numpy(events["p"].astype(np.int8))
            
          return vis.shist(x, y, t, p, ch, height, width).to(device)

def vtei(events, height, width, ch, device):
          x = torch.from_numpy(np.clip(events["x"].astype(np.int16).copy(),0,width-1))
          y = torch.from_numpy(np.clip(events["y"].astype(np.int16).copy(),0,height-1))
          t = torch.from_numpy(events["t"].astype(np.int64))
          p  = torch.from_numpy(events["p"].astype(np.int8))
            
          return  vis.vtei(x, y, t, p, ch, height, width).to(device)

def taf(events, height, width, ch, device):
        event_window_abin = 10000
        events_window = event_window_abin*ch
        start_time = int(events["t"][0])
        end_time = int(events["t"][-1])  

        if (end_time - start_time) < events_window:
                    start_time = end_time - events_window
        else:
                    start_time = end_time - round((end_time - start_time - events_window)/event_window_abin) * event_window_abin - events_window

        t_ = torch.from_numpy(events["t"].copy())
        z = torch.zeros_like(t_)

        bins = math.ceil((end_time - start_time) / event_window_abin)
                
        for i in range(bins):
                z = torch.where((t_ >= start_time + i * event_window_abin)&(t_ <= start_time + (i + 1) * event_window_abin), torch.zeros_like(t_)+i, z)
                
      
        memory = torch.zeros(height, width, 2, ch) - 6000
        
        for iter in range(bins):
                events_ = events[z == iter]
                t_max = start_time + (iter + 1) * event_window_abin
                t_min = start_time + iter * event_window_abin
                events_["t"] = (events_["t"] - t_min)/(t_max - t_min + 1e-8)
                x = torch.from_numpy(np.clip(events_["x"].astype(np.int64).copy(),0,width-1))
                y = torch.from_numpy(np.clip(events_["y"].astype(np.int64).copy(),0,height-1))
                t = torch.from_numpy(events_["t"].astype(np.int64))
                p  = torch.from_numpy(events_["p"].astype(np.int64).copy())
                img, memory, generate_time = vis.taf_cuda(x.long(), y.long(), t.float(), p.long(),  [height, width], ch, memory) 

        img = vis.leaky_transform(img)
        return img.to(device)

def kaiming_normal_fanin_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        if m.affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

def kaiming_normal_fanout_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        if m.affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

def kaiming_uniform_fanin_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        if m.affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

def kaiming_uniform_fanout_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        if m.affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

def xavier_normal_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        if m.affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

def xavier_uniform_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        if m.affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

def plain_normal_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.1)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        if m.affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

def plain_uniform_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.uniform_(m.weight, a=-0.1, b=0.1)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        if m.affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

def init_model(model, method='kaiming_norm_fanin'):
    if method == 'kaiming_norm_fanin':
        model.apply(kaiming_normal_fanin_init)
    elif method == 'kaiming_norm_fanout':
        model.apply(kaiming_normal_fanout_init)
    elif method == 'kaiming_uni_fanin':
        model.apply(kaiming_uniform_fanin_init)
    elif method == 'kaiming_uni_fanout':
        model.apply(kaiming_uniform_fanout_init)
    elif method == 'xavier_norm':
        model.apply(xavier_normal_init)
    elif method == 'xavier_uni':
        model.apply(xavier_uniform_init)
    elif method == 'plain_norm':
        model.apply(plain_normal_init)
    elif method == 'plain_uni':
        model.apply(plain_uniform_init)
    else:
        raise NotImplementedError
    return model



def compute_aznas_score(model, gpu, event_data, files, encoding, ch, actual_res, target_res, batch_size, init_method = 'kaiming_norm_fanin', fp16=False):
    #https://github.com/cvlab-yonsei/AZ-NAS/blob/5e6683a2cfa5c6d0dc34a1317a842497ba7eae47/NB201/ZeroShotProxy/compute_az_nas_score.py#L109
    model.train()
    model.cuda()
    info = {}
    nas_score_list = []
    if gpu is not None:
        device = torch.device('cuda:{}'.format(gpu))
    else:
        device = torch.device('cpu')

    if fp16:
        dtype = torch.half
    else:
        dtype = torch.float32

    init_model(model, init_method)

    if not event_data:
        resolution = target_res

        input_ = torch.randn(size=[batch_size, ch, resolution[0], resolution[1]], device=device, dtype=dtype)
    else:
          if event_data:
             input_ = []
             for i in range(batch_size):
              if encoding in ["mdes","vtei"]: 
                input_.append(create_event_input(files[i +(batch_size)], target_res, actual_res, ch, encoding, device=device, dtype=dtype))
              else: 
                input_.append(create_event_input(files[i +(batch_size)], target_res, actual_res, ch // 2, encoding, device=device, dtype=dtype))
             input_ = torch.stack(input_, dim =0)


    hidden_states = {"0": torch.empty(0), "1": torch.empty(0), "2": torch.empty(0), "3": torch.empty(0)}
    layer_features = model.extract_cell_features([input_, hidden_states])

    ################ expressivity & progressivity scores ################
    expressivity_scores = []
    for i in range(len(layer_features)):
        feat = layer_features[i].detach().clone()
        b,c,h,w = feat.size()
        feat = feat.permute(0,2,3,1).contiguous().view(b*h*w,c)
        m = feat.mean(dim=0, keepdim=True)
        feat = feat - m
        sigma = torch.mm(feat.transpose(1,0),feat) / (feat.size(0))
        s = torch.linalg.eigvalsh(sigma) # faster version for computing eignevalues, can be adopted since sigma is symmetric
        prob_s = s / s.sum()
        score = (-prob_s)*torch.log(prob_s+1e-8)
        score = score.sum().item()
        expressivity_scores.append(score)
    expressivity_scores = np.array(expressivity_scores)
    progressivity = np.min(expressivity_scores[1:] - expressivity_scores[:-1])
    expressivity = np.sum(expressivity_scores)
    #####################################################################

    ################ trainability score ##############
    scores = []
    for i in reversed(range(1, len(layer_features))):
        f_out = layer_features[i]
        f_in = layer_features[i-1]
        if f_out.grad is not None:
            f_out.grad.zero_()
        if f_in.grad is not None:
            f_in.grad.zero_()
        
        g_out = torch.ones_like(f_out) * 0.5
        g_out = (torch.bernoulli(g_out) - 0.5) * 2
        g_in = torch.autograd.grad(outputs=f_out, inputs=f_in, grad_outputs=g_out, retain_graph=False)[0]
        if g_out.size()==g_in.size() and torch.all(g_in == g_out):
            scores.append(-np.inf)
        else:
            if g_out.size(2) != g_in.size(2) or g_out.size(3) != g_in.size(3):
                bo,co,ho,wo = g_out.size()
                bi,ci,hi,wi = g_in.size()
                stride = int(hi/ho)
                pixel_unshuffle = nn.PixelUnshuffle(stride)
                g_in = pixel_unshuffle(g_in)
            bo,co,ho,wo = g_out.size()
            bi,ci,hi,wi = g_in.size()
            ### straight-forward way
            # g_out = g_out.permute(0,2,3,1).contiguous().view(bo*ho*wo,1,co)
            # g_in = g_in.permute(0,2,3,1).contiguous().view(bi*hi*wi,ci,1)
            # mat = torch.bmm(g_in,g_out).mean(dim=0)
            ### efficient way # print(torch.allclose(mat, mat2, atol=1e-6))
            g_out = g_out.permute(0,2,3,1).contiguous().view(bo*ho*wo,co)
            g_in = g_in.permute(0,2,3,1).contiguous().view(bi*hi*wi,ci)
            mat = torch.mm(g_in.transpose(1,0),g_out) / (bo*ho*wo)
            ### make it faster
            if mat.size(0) < mat.size(1):
                mat = mat.transpose(0,1)
            ###
            s = torch.linalg.svdvals(mat)
            scores.append(-s.max().item() - 1/(s.max().item()+1e-6)+2)
    trainability = np.mean(scores)
    #################################################

    info['expressivity'] = float(expressivity) if not np.isnan(expressivity) else -np.inf
    info['progressivity'] = float(progressivity) if not np.isnan(progressivity) else -np.inf
    info['trainability'] = float(trainability) if not np.isnan(trainability) else -np.inf
    # info['complexity'] = float(model.get_FLOPs(resolution)) # take info from api
    return info

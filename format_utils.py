from __future__ import print_function
import numpy as np
import torch
import math
import time


def taf_cuda(x, y, t, p, shape, volume_bins, past_volume):
    # from https://github.com/HarmoniaLeo/FRLW-EvD/blob/66fa1b7b53399bf4534d10dc81a6db457bc62bbb/generate_taf.py#L18
    tick = time.time()
    H, W = shape

    img = torch.zeros((H * W * 2)).float().to(x.device)
    img.index_add_(0, p + 2 * x + 2 * W * y, torch.ones_like(x).float())
    t_img = torch.zeros((H * W * 2)).float().to(x.device)
    t_img.index_add_(0, p + 2 * x + 2 * W * y, t - 1.0)
    t_img = t_img/(img+1e-8)

    img = img.view(H, W, 2)
    t_img = t_img.view(H, W, 2)
    torch.cuda.synchronize()
    generate_volume_time = time.time() - tick

    tick = time.time()
    forward = (img == 0)
    torch.cuda.synchronize()
    filter_time = time.time() - tick
    tick = time.time()
    old_ecd = past_volume
    if torch.all(forward):
        ecd = old_ecd
    else:
        ecd = t_img[:, :, :, None]
        ecd = torch.cat([old_ecd, ecd],dim=3)
        for i in range(1,ecd.shape[3])[::-1]:
            ecd[:,:,:,i-1] = ecd[:,:,:,i-1] - 1
            ecd[:,:,:,i] = torch.where(forward, ecd[:,:,:,i-1],ecd[:,:,:,i])
        if ecd.shape[3] > volume_bins:
            ecd = ecd[:,:,:,1:]
        else:
            ecd[:,:,:,0] = torch.where(forward, torch.zeros_like(forward).float() -6000, ecd[:,:,:,0])
    torch.cuda.synchronize()
    generate_encode_time = time.time() - tick

    ecd_viewed = ecd.permute(3, 2, 0, 1).contiguous().view(volume_bins * 2, H, W)

    #print(generate_volume_time, filter_time, generate_encode_time)
    return ecd_viewed, ecd, generate_encode_time + generate_volume_time
    
def leaky_transform(ecd):
    ### used for TAF
    ecd = ecd.clone()
    ecd = torch.log1p(-ecd)
    ecd = 1 - ecd / 8.7
    ecd = torch.where(ecd < 0, torch.zeros_like(ecd), ecd)
    ecd = ecd * 255
    return ecd
   
    
def shist(x,y,t,p, bins, height, width, device = "cpu"):
    # https://github.com/uzh-rpg/RVT/blob/master/data/utils/representations.py#L124
    dtype = torch.uint8
    #assert p.min() >= 0
    #assert p.max() <= 1
    representation = torch.zeros((2,bins, height, width), dtype=dtype, device=device, requires_grad=False)
    t0 = t[0]
    t1 = t[-1]
    
    tnorm = t - t0
    tnorm = tnorm/ max((t1-t0),1)
    tnorm = tnorm*bins
    t_idx = tnorm.floor()
    t_idx = torch.clamp(t_idx, max = bins - 1)

    indices = x.long() + \
                  width * y.long() + \
                  height *  width  * t_idx.long() + \
                  bins * height * width * p.long()
    values = torch.ones_like(indices, dtype=dtype, device=device)
    representation.put_(indices, values, accumulate=True)
    representation = torch.clamp(representation, min=0, max=255)

    return torch.reshape(representation, (-1, height, width))


def voxel_grid(x,y,t,p, bins, height, width, device = "cpu"):
    # https://github.com/uzh-rpg/RVT/blob/master/data/utils/representations.py#L124
    dtype = torch.half
    #assert p.min() >= 0
    #assert p.max() <= 1

    
    representation = torch.zeros((2,bins, height, width), dtype=dtype)
    t0 = t[0]
    t1 = t[-1]
   
    tnorm = t - t0
    
    tnorm = tnorm/max((t1-t0),1)
    tnorm = tnorm*bins
    t_idx = tnorm.floor()
    t_idx = torch.clamp(t_idx, max = bins - 1)
    values = torch.maximum(torch.zeros_like(tnorm, dtype= dtype), 1 - torch.abs(tnorm - t_idx)).to(dtype=dtype)

    

    indices = x.long() + \
                  width * y.long() + \
                  height *  width  * t_idx.long() + \
                  bins * height * width * p.long()
    
    representation.put_(indices, values, accumulate=True)
    return torch.reshape(representation, (-1, height, width))

def ev_temporal_volume(x,y,t,p, bins, height, width, device = "cpu"):
    # https://github.com/uzh-rpg/RVT/blob/master/data/utils/representations.py#L124
    dtype = torch.int16
    #assert p.min() >= 0
    #assert p.max() <= 1
    representation = torch.zeros((bins, height, width), dtype=dtype, device=device, requires_grad=False)
    t0 = t[0]
    t1 = t[-1]
    
    p = 2*p - 1

    tnorm = t - t0
    tnorm = tnorm/ max((t1-t0),1)
    tnorm = tnorm*bins
    t_idx = tnorm.floor()
    t_idx = torch.clamp(t_idx, max = bins - 1)
    
    indices = x.long() + width*y.long() + height*width*t_idx.long()

   

    values = torch.asarray(p, dtype=dtype, device = device)
    
    
    
    representation.put_(indices, values, accumulate=True)

    
    #representation = torch.clamp(representation, min=-1, max=1)
   
    return torch.reshape((255.0/( 1 + torch.exp(-representation/2))).to(dtype=torch.uint8), (-1, height, width))


def vtei(x,y,t,p, bins, height, width, device = "cpu"):
    # https://github.com/uzh-rpg/RVT/blob/master/data/utils/representations.py#L124
    dtype = torch.int8
    #assert p.min() >= 0
    #assert p.max() <= 1

    representation = torch.zeros((bins, height, width), dtype=dtype, device=device, requires_grad=False)
    t0 = t[0]
    t1 = t[-1]
    
    p = 2*p - 1

    tnorm = t - t0
    tnorm = tnorm/ max((t1-t0),1)
    tnorm = tnorm*bins
    t_idx = tnorm.floor()
    t_idx = torch.clamp(t_idx, max = bins - 1)
    
    indices = x.long() + width*y.long() + height*width*t_idx.long()

   

    values = torch.asarray(p, dtype=dtype, device = device)
    
    
    
    representation.put_(indices, values, accumulate=False)

    
    #representation = torch.clamp(representation, min=-1, max=1)
   
    return torch.reshape(representation, (-1, height, width))

def mdes(x,y,t,p, bins, height, width, device = "cpu"):
    # https://github.com/uzh-rpg/RVT/blob/master/data/utils/representations.py#L124
    dtype = torch.int8
    representation = torch.zeros((bins, height, width), dtype=dtype, device=device, requires_grad=False)
    
    p = 2*p - 1

    t0 = t[0]
    t1 = t[-1]
    tnorm = (t - t0)/ max((t1-t0),1)
    tnorm = torch.clamp(tnorm, min=1e-6, max = 1 - 1e-6)
    bin_float = bins - torch.log(tnorm)/ math.log(1/2)
    bin_float = torch.clamp(bin_float, min = 0)
    t_idx = bin_float.floor()
    indices = x.long() + width*y.long() + height*width*t_idx.long()
    
    values = torch.asarray(p, dtype=dtype, device = device)
    representation.put_(indices, values, accumulate=True)
    
    

    for i in reversed(range(bins)):
        representation[i] = torch.sum(input=representation[:i + 1], dim=0)


    return representation


import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch import nn
import numpy as np
import  argparse,time
import yaml

from ultralytics.nn.tasks import DetectionModel2


from ModelDebug import NAS_MEASURE_MODEL

def network_weight_gaussian_init(net: nn.Module):
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            else:
                continue

    return net

def compute_nas_score(gpu, model, ch, mixup_gamma, resolution, batch_size, repeat, fp16=False):
    # https://github.com/idstcv/ZenNAS/blob/d1d617e0352733d39890fb64ea758f9c85b28c1a/ZeroShotProxy/compute_zen_score.py#L33
    info = {}
    nas_score_list = []
    if gpu is not None:
        device = torch.device(gpu)
    else:
        device = torch.device('cpu')

    if fp16:
        dtype = torch.half
    else:
        dtype = torch.float32

    with torch.no_grad():
        for repeat_count in range(repeat):
            network_weight_gaussian_init(model)
            input = torch.randn(size=[batch_size, ch, resolution[0], resolution[1]], device=device, dtype=dtype)
            input2 = torch.randn(size=[batch_size, ch, resolution[0], resolution[1]], device=device, dtype=dtype)
            mixup_input = input + mixup_gamma * input2
            
            h1 = {"0":torch.empty(0), "1":torch.empty(0), "2":torch.empty(0), "3": torch.empty(0)}
            h2 = {"0":torch.empty(0), "1":torch.empty(0), "2":torch.empty(0), "3": torch.empty(0)}
          
            output, h1 = model(input,h1)
            
            mixup_output, h2 = model(mixup_input,h2)

            nas_score = torch.sum(torch.abs(output - mixup_output), dim=[1, 2, 3])
            nas_score = torch.mean(nas_score)

            # compute BN scaling
            log_bn_scaling_factor = 0.0
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    bn_scaling_factor = torch.sqrt(torch.mean(m.running_var))
                    log_bn_scaling_factor += torch.log(bn_scaling_factor)
                pass
            pass
            nas_score = torch.log(nas_score) + log_bn_scaling_factor
            nas_score_list.append(float(nas_score))


    std_nas_score = np.std(nas_score_list)
    avg_precision = 1.96 * std_nas_score / np.sqrt(len(nas_score_list))
    avg_nas_score = np.mean(nas_score_list)


    info['avg_nas_score'] = float(avg_nas_score)
    info['std_nas_score'] = float(std_nas_score)
    info['avg_precision'] = float(avg_precision)
    return info


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16, help='number of instances in one mini-batch.')
    parser.add_argument('--input_image_size', type=int, default=None,
                        help='resolution of input image, usually 32 for CIFAR and 224 for ImageNet.')
    parser.add_argument('--repeat_times', type=int, default=32)
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--mixup_gamma', type=float, default=1e-2)
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

if __name__ == "__main__":
 
    args = parse_opt()
    cfg = "model_90l.yaml"
    gpu = "cuda:0"
    #cfg = "ReYOLOV8s.yaml"
    the_model = NAS_MEASURE_MODEL(cfg)
    if gpu is not None:
        the_model = the_model.cuda(gpu)


    start_timer = time.time()
    info = compute_nas_score(gpu=gpu, ch = 10, model=the_model, mixup_gamma=args.mixup_gamma,
                             resolution=[256,320], batch_size=args.batch_size, repeat=args.repeat_times, fp16=False)
    time_cost = (time.time() - start_timer) / args.repeat_times
    zen_score = info['avg_nas_score']
    print(f'zen-score={zen_score:.4g}, time cost={time_cost:.4g} second(s)')
    
    
    


import torch
import argparse
from models import *

@torch.no_grad()
def evaluate_time(Net,imgL,imgR,device,**kwargs):
    import time

    for i in range(10):
        preds = Net(imgL, imgR)

    times = 30
    start = time.perf_counter()
    for i in range(times):
        preds = Net(imgL, imgR)
    end = time.perf_counter()

    avg_run_time = (end - start) / times

    return avg_run_time

@torch.no_grad()
def evaluate_flops(Net,input,device,**kwargs):

    from fvcore.nn import FlopCountAnalysis
    flops = FlopCountAnalysis(Net,input)   # FLOPs（乘加=2）
    total_flops = flops.total()

    total_params = sum(p.numel() for p in Net.parameters())
    # print(f"\nFLOPs: {total_flops/1e9:.2f} GFLOPs, parameters: {total_params / 1e6:.2f} M")

    return total_flops,total_params

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--maxdisp', type=int ,default=192,
                        help='maxium disparity')

    parser.add_argument('-batch_size', default=1, type=int)

    parser.add_argument('-mixup_alpha', default=0.5, type=float)
    parser.add_argument('-grad_clip_value', default=1., type=float)
    parser.add_argument('-device', default='cuda', type=str) #cuda:0, cpu

    args = parser.parse_args()


    #model
    Net = stackhourglass(args.maxdisp)

    imgL = torch.randn(1,3,544,960)
    imgR = torch.randn(1,3,544,960)

    Net = Net.to(args.device)
    imgL = imgL.to(args.device)
    imgR = imgR.to(args.device)

    avg_run_time = evaluate_time(Net=Net,imgL=imgL,imgR=imgR,device=args.device)
    total_flops,total_params = evaluate_flops(Net,input=(imgL,imgL),device=args.device)

    print(avg_run_time)
    print(f"\nFLOPs: {total_flops/1e9:.2f} GFLOPs, parameters: {total_params / 1e6:.2f} M")
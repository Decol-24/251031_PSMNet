import torch
import argparse
from models import *

@torch.no_grad()
def evaluate_time(Net, imgL, imgR, device, warmup=30, times=50):
    Net = Net.to(device).eval()
    imgL = imgL.to(device)
    imgR = imgR.to(device)

    # warmup
    for _ in range(warmup):
        with torch.amp.autocast('cuda', enabled=True):
            _ = Net(imgL, imgR)
    torch.cuda.synchronize()

    starter = torch.cuda.Event(enable_timing=True)
    ender   = torch.cuda.Event(enable_timing=True)

    total_ms = 0.0
    for _ in range(times):
        starter.record()
        with torch.amp.autocast('cuda', enabled=True):
            _ = Net(imgL, imgR)
        ender.record()
        torch.cuda.synchronize()
        total_ms += starter.elapsed_time(ender)

    avg_s = (total_ms / times) / 1000.0
    return avg_s

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
import math
import torch
import time

def UpdateMultiTaskWeightWithAlphas(FM, alphas):
    keys = list(FM[0].keys())
    total = {key: torch.zeros_like(FM[0][key]) for key in keys}
    Lw = [{key: torch.zeros_like(FM[0][key]) for key in keys} for i in range(len(alphas))]

    for i, alpha in enumerate(alphas):
        total = {key: total[key] + alpha * FM[i][key] for key in keys}

    for i in range(len(alphas)):
        Lw[i] = {key: Lw[i][key] + alphas[i] * FM[i][key] / total[key] for key in keys}
    for i in range(len(alphas)):
        for key in keys:
            if not torch.isfinite(Lw[i][key].all()):
                print(i,key)
                exit()
    return Lw

def AddMultiTaskLayers(L_copy, model, Lw, alphas):
    keys = list(Lw[0].keys())
    for key in keys:
        w = 0
        # time.sleep(3)
        for l in range(len(alphas)):
            w += L_copy[l][key].to(torch.float32) * Lw[l][key]
            # print(Lw[l][key].to(torch.float16))
            
        with torch.no_grad():
            for n, p in model.named_parameters():
                if key == n:
                    p.copy_(w.to(torch.float16))
                    break
        
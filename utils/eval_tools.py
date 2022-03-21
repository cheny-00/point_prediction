import torch


def ade(pred, target):
    pred, target = pred.clone().detach().cpu(), target.clone().detach().cpu()
    dist = torch.sum(torch.sqrt(torch.sum(torch.pow(pred - target, 2), -1))) / len(pred)
    return dist

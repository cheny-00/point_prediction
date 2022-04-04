import math
import torch

def do_eval_step(func):
    def wrapper(*args, **kwargs):
        move_to_cpu = lambda x: x.clone().detach().cpu()
        return func(*(move_to_cpu(arg) for arg in args), **kwargs)
    return wrapper

@do_eval_step
def ade(pred, target):
    dist = torch.sum(torch.sqrt(torch.sum(torch.pow(pred - target, 2), -1))) / len(pred)
    return dist

@do_eval_step
def rmsd(pred, target):
    loss = torch.mean(torch.sum(torch.pow(pred - target, 2), -1))
    return loss

@do_eval_step
def aae(prev, pred, label):
    cal_dist = lambda x, y: torch.sqrt(torch.sum(torch.pow(x - y, 2), -1))
    start = prev[:, -1, :2]
    a = cal_dist(label, start)
    b = cal_dist(start, pred)
    c = cal_dist(label, pred)
    if a * b == 0:
        return torch.arccos(0)
    cos_c = (torch.pow(a, 2) + torch.pow(b, 2) + torch.pow(c, 2)) / (2 * a * b)
    return torch.mean(torch.arccos(torch.clip(cos_c, -0.99999, 0.99999)))


def normalize_label(pred_time, targets):
    time_indent = 1000 / 240
    n_points = targets.shape[1] - 1
    idx = pred_time / time_indent - 1
    lower_idx, upper_idx = int(math.floor(idx)), int(math.ceil(idx))
    if lower_idx < 0 or upper_idx <=0:
        return targets[:, 0, :]
    if lower_idx >= n_points or upper_idx > n_points:
        return targets[:, n_points - 1, :]
    alpha = (pred_time - lower_idx * time_indent) / time_indent
    beta = 1 - alpha
    norm_label = alpha * targets[:, lower_idx, :] + beta * targets[:, upper_idx, :]
    return norm_label

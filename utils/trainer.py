import os
import math
from time import time
import torch
import torch.nn as nn
from tqdm import tqdm
from functools import partial
from utils import eval_tools

class Trainer:

    def __init__(self,
                 epochs,
                 train_iter,
                 dev_iter,
                 logger,
                 batch_size,
                 log_interval,
                 eval_interval,
                 optim,
                 save_path,
                 is_qat,
                 scheduler):

        self.epochs = epochs
        self.train_iter = train_iter
        self.dev_iter = dev_iter
        self.logger = logger
        self.batch_size = batch_size
        self.log_interval = log_interval
        self.optim = optim
        self.eval_interval = eval_interval
        self.save_path = save_path
        self.is_qat = is_qat
        self.scheduler = scheduler



    def train(self,
              model,
              crit):

        train_process = partial(self.train_process, model=model)

        total_proj_angle_loss = total_dist_loss = total_fit_loss = total_angle_loss = total_pred_time = total_mse = train_loss = train_step = 0
        log_interval, eval_interval = self.log_interval, self.eval_interval

        device = next(model.parameters()).device
        model.train()
        best_score = float('inf')
        for i in range(self.epochs):
            tqdm_train_iter = tqdm(self.train_iter)
            log_start_time = time()

            for num_batch, data in enumerate(tqdm_train_iter):
                logits, loss, label, loss_details = train_process(data=data)

                total_mse += torch.mean(loss_details[0]).float().item()
                total_angle_loss += torch.mean(loss_details[1]).float().item()
                total_pred_time += loss_details[2].float().item()
                total_fit_loss += loss_details[3].float().item()
                total_dist_loss += loss_details[4].float().item()
                total_proj_angle_loss += loss_details[5].float().item()

                train_loss += loss.float().item()
                # TODO will we use fp16?
                loss.backward()
                self.optim.step()
                train_step += 1
                # nn.utils.clip_grad_norm(model.parameters(), 10.0) #TODO clip?

                if train_step % log_interval == 0:

                    cur_loss = train_loss / log_interval
                    elapsed = time() - log_start_time
                    log_str = f"| epoch {i} steps {train_step} | {num_batch} batches | lr {self.optim.param_groups[0]['lr']:.3g} " \
                              f"| {elapsed * 1000 / log_interval:5.2f} ms/batch | mse_loss {math.sqrt(total_mse / log_interval):5.6f}| loss {cur_loss:5.6f} | predicted_time {total_pred_time / log_interval:3.6f} " \
                              f"| dist loss:{total_dist_loss / log_interval:5.6f} | fit_loss: {total_fit_loss / log_interval:5.6f} | proj angle loss:{total_proj_angle_loss / log_interval:5.6f}"

                    self.logger(log_str, print_=False)
                    tqdm_train_iter.set_description(log_str, refresh=True)
                    total_proj_angle_loss = total_dist_loss = total_fit_loss = total_angle_loss = total_pred_time = total_mse = train_loss = 0
                    log_start_time = time()
            if (i + 1) % eval_interval == 0:

                dev_start_time = time()
                scores = self.eval_model(model, train_step) # TODO should we add early stop?
                if type(scores) != type(dict()):
                    rmsd_loss, angle_loss, valid_score, pred_time = 0, 0, scores, 0
                else:
                    rmsd_loss, angle_loss, valid_score, pred_time =\
                        scores['rmsd_loss'], scores['angle_loss'], scores['valid_score'], scores['pred_time']
                log_str = f"| Eval step at {train_step} | speed time: {time() - dev_start_time} |" \
                          f"| rmsd: {rmsd_loss:5.4f} | angle loss: {angle_loss:5.4f} | average valid loss: {valid_score:5.7f} | predicted time: {pred_time:4.6f} "
                self.logger('-' * len(log_str) + "\n" + log_str + "\n" + '-' * len(log_str), print_=True)

                if (i + 1) > 60 and (valid_score < best_score or (i + 1) in [self.epochs// 3, self.epochs // 2, self.epochs]):
                    if self.is_qat:
                        model_int8 = torch.quantization.convert(model)
                        model_int8.cpu()
                        torch.save({"model_state_dict": model_int8.state_dict()},
                                   os.path.join(self.save_path, f"epoch_{(i+1)}_{valid_score:5.3f}.pt"))
                        model.to(device)
                        continue
                    model.cpu()
                    torch.save({"model_state_dict": model.state_dict()},
                               os.path.join(self.save_path, f"epoch_{(i+1)}_{valid_score:5.3f}.pt"))
                    model.to(device)
                    best_score = valid_score
            self.scheduler.step()


    def train_process(self, *args, **kwargs):
        return NotImplementedError()

    def eval_model(self, *args, **kwargs):
        return NotImplementedError()



class LSTMModelTrainer(Trainer):

    def train_process(self, model, data, **kwargs):

        self.optim.zero_grad()
        logits, loss = model(data)
        return logits, loss, data['label'], 1

    def eval_model(self,
                   model,
                   train_step):
        model.eval()
        tqdm_dev_iter = tqdm(self.dev_iter)
        dev_total_loss = 0

        with torch.no_grad():
            for data in tqdm_dev_iter:
                _, dev_loss = model(data)
                dev_total_loss += dev_loss

        model.train()
        return dev_total_loss / len(tqdm_dev_iter)




class DepressModelTrainer(Trainer):
    def train_process(self, model, data, **kwargs):

        self.optim.zero_grad()
        logits, loss, loss_details = model(data)
        # norm_label = eval_tools.normalize_label(pred_time, data['targets'])
        norm_label = data['targets'][:, -1, :]
        return logits, loss, norm_label, loss_details
    
    def eval_model(self, model, train_step):
        return self.eval_step(model, self.dev_iter)
    
    @staticmethod
    def eval_step(model, dev_iter):
        model.eval()
        tqdm_dev_iter = tqdm(dev_iter)
        n = len(tqdm_dev_iter)
        total_pred_time = total_loss = total_angle_loss = total_ade_loss = total_rmsd_loss = 0

        with torch.no_grad():
            for data in tqdm_dev_iter:
                pred, loss, loss_details  = model(data)
                # norm_label = eval_tools.normalize_label(pred_time, data['targets'])
                norm_label = data['targets'][:, -1, :]

                ade_loss = eval_tools.ade(pred, norm_label)
                angle_loss = eval_tools.aae(data['input'], pred, norm_label)


                total_rmsd_loss += torch.mean(loss_details[0]).float().item()
                total_angle_loss += torch.mean(loss_details[1]).float().item()
                total_ade_loss += ade_loss.item()
                total_pred_time += (1 - loss_details[2].float().item())
                total_loss += loss.item()

        model.train()
        ret = {
            'rmsd_loss': math.sqrt(total_rmsd_loss / n),
            'angle_loss': total_angle_loss / n,
            'ade_loss': total_ade_loss / n,
            'valid_score': total_loss / n,
            'pred_time': total_pred_time / n
        }
        return ret

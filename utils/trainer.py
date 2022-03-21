import os
from time import time
import torch
import torch.nn as nn
from tqdm import tqdm
from functools import partial

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
                 is_qat):

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



    def train(self,
              model,
              crit):

        train_process = partial(self.train_process, model=model)

        train_loss = train_step = 0
        log_interval, eval_interval = self.log_interval, self.eval_interval

        device = next(model.parameters()).device
        model.train()
        best_score = float('inf')
        for i in range(self.epochs):
            tqdm_train_iter = tqdm(self.train_iter)
            log_start_time = time()

            for num_batch, data in enumerate(tqdm_train_iter):
                loss = train_process(data=data)
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
                              f"| {elapsed * 1000 / log_interval:5.2f} ms/batch | loss {cur_loss:5.6f}"
                    self.logger(log_str, print_=False)
                    tqdm_train_iter.set_description(log_str, refresh=True)
                    train_loss = 0
                    log_start_time = time()
            if (i + 1) % eval_interval == 0:
                valid_score = self.eval_model(model, train_step) # TODO should we add early stop?
                if (i + 1) > 150 and (valid_score < best_score or (i + 1) in [self.epochs// 3, self.epochs // 2, self.epochs]):
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




    def train_process(self, *args, **kwargs):
        return NotImplementedError()
    def eval_model(self, *args, **kwargs):
        return NotImplementedError()


class LSTMModelTrainer(Trainer):

    def train_process(self, model, data, **kwargs):

        self.optim.zero_grad()
        _, loss = model(data)
        return loss

    def eval_model(self,
                   model,
                   train_step):
        model.eval()
        tqdm_dev_iter = tqdm(self.dev_iter)
        dev_total_loss, dev_start_time = 0, time()

        with torch.no_grad():
            for data in tqdm_dev_iter:
                _, dev_loss = model(data)
                dev_total_loss += dev_loss


        log_str  = f"| Eval step at {train_step} | speed time: {time() - dev_start_time} |" \
                   f"| average valid loss: {dev_total_loss / len(tqdm_dev_iter)} | "

        self.logger('-' * 100 + "\n" + log_str + "\n" + '-' * 100, print_=False)
        tqdm_dev_iter.set_description(log_str)

        model.train()
        return dev_total_loss / len(tqdm_dev_iter)


import os
from time import time
import torch
import torch.nn as nn
from tqdm import tqdm
from functools import partial

class Trainer:

    def __init__(self,
                 epoches,
                 train_iter,
                 dev_iter,
                 logger,
                 batch_size,
                 log_interval,
                 optim):

        self.epoches = epoches
        self.train_iter = train_iter
        self.dev_iter = dev_iter
        self.logger = logger
        self.batch_size = batch_size
        self.log_interval = log_interval
        self.optim = optim



    def train(self,
              model,
              crit):

        device = next(model.parameters()).device
        train_process = partial(self.train_process, crit=crit)

        train_loss = train_step = 0
        log_interval = self.log_interval

        for i in range(self.epoches):
            tqdm_train_iter = self.train_iter
            log_start_time = time()

            for num_batch, data in enumerate(tqdm_train_iter):
                loss = train_process(data)
                train_loss += loss.float().item()
                # TODO will we use fp16?
                loss.backward()
                train_step += 1
                nn.utils.clip_grad_norm(model.parameters(), 10.0) #TODO clip?

                if train_step % log_interval == 0:

                    cur_loss = train_loss / log_interval
                    elapsed = time() - log_start_time
                    log_str = f"| epoch {i} steps {train_step} | {num_batch} batchs | lr {self.optim.param_groups[0]['lr']:.3g} " \
                              f"| {elapsed * 1000 / log_interval:5.2f} ms/batch | loss {cur_loss:5.6f}"
                    self.logger(log_str, print_=False)
                    tqdm_train_iter.set_description(log_str, refresh=False)
                    train_loss = 0
                    log_start_time = time()




    def train_process(self, *args, **kwargs):
        return NotImplementedError()
    def eval_model(self, *args, **kwargs):
        return NotImplementedError()


class SamsungModelTrainer(Trainer):

    @classmethod
    def train_process(cls, data, **kwargs):

        self.optim.zero_grad()
        loss = cls.model(data)
        return loss

    def eval_model(self,
                   ):
        pass




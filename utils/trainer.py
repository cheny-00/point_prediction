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
                 optim):

        self.epochs = epochs
        self.train_iter = train_iter
        self.dev_iter = dev_iter
        self.logger = logger
        self.batch_size = batch_size
        self.log_interval = log_interval
        self.optim = optim
        self.eval_interval = eval_interval



    def train(self,
              model,
              crit):

        train_process = partial(self.train_process, crit=crit)

        train_loss = train_step = 0
        log_interval, eval_interval = self.log_interval, self.eval_interval

        model.train()
        for i in range(self.epochs):
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
                    log_str = f"| epoch {i} steps {train_step} | {num_batch} batches | lr {self.optim.param_groups[0]['lr']:.3g} " \
                              f"| {elapsed * 1000 / log_interval:5.2f} ms/batch | loss {cur_loss:5.6f}"
                    self.logger(log_str, print_=False)
                    tqdm_train_iter.set_description(log_str, refresh=False)
                    train_loss = 0
                    log_start_time = time()
            if (i + 1) % eval_interval:
                self.eval_model(model, train_step)



    def train_process(self, *args, **kwargs):
        return NotImplementedError()
    def eval_model(self, *args, **kwargs):
        return NotImplementedError()


class LSTMModelTrainer(Trainer):

    def train_process(self, data, **kwargs):

        self.optim.zero_grad()
        _, loss = self.model(data)
        return loss

    def eval_model(self,
                   model,
                   train_step):
        model.eval()
        tqdm_dev_iter = self.dev_iter
        total_loss, eval_start_time = 0, time()

        with torch.no_grad():
            for data in enumerate(tqdm_dev_iter):
                _, loss = model(data)


        log_str  = f"| Eval step at {train_step} | speed time: {time() - eval_start_time} |" \
                   f"| average valid loss: {total_loss / len(tqdm_dev_iter)} | "

        self.logger('-' * 100 + "\n" + log_str + "\n" + '-' * 100, print_=False)
        tqdm_dev_iter.set_description(log_str)

        model.train()


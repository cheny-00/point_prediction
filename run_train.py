import os
import time

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from load_args import load_args
from utils.data import LSTMDataset
from utils.utils import create_exp_dir, randn_sampler
from utils.trainer import LSTMModelTrainer, DepressModelTrainer
from model.model_table import *

#############################################################################################
##  setting
#############################################################################################
args = load_args()

start_time = time.strftime('%Y%m%d-%H%M%S')
st_date, st_time = start_time.split("-")
work_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
log_dir = os.path.join(work_dir, 'logs', args.proj_name, args.model_name, st_date, st_time)
working_files  = ["run_train.py", "load_args.py", f"model/{args.model_name}_model.py", "utils/trainer.py"]
working_files = list(map(lambda x: os.path.join(work_dir, x), working_files))
logging = create_exp_dir(log_dir,
                         scripts_to_save=working_files,
                         debug=args.debug)

device = torch.device(f'cuda:{args.rank}' if args.cuda else 'cpu')
if torch.cuda.is_available() and not args.cuda:
    print("detected gpu is available but not used")


save_path = os.path.join(work_dir, "save", args.model_name, st_date, st_time)
if not os.path.exists(save_path) and not args.debug:
    os.makedirs(save_path)
#############################################################################################
##  load dataset
#############################################################################################
val_split = [0.15, 0.1]


dataset = LSTMDataset(data_path=args.data_path,
                      offset=args.offset,
                      use_n_data=args.n_data)

dataset.get_data()
# print(dataset.dt)
#if not os.path.exists(args.load_dataset_path):
#    torch.save(dataset, args.data)
dev_sampler, eval_sampler, train_sampler = randn_sampler(val_split,
                                                         len(dataset),
                                                         shuffle_dataset=True,
                                                         random_seed=args.seed)


train_iter = DataLoader(dataset,
                        batch_size=args.batch_size,
                        drop_last=False,
                        sampler=train_sampler)
dev_iter = DataLoader(dataset,
                      batch_size=args.eval_batch_size,
                      sampler=dev_sampler)
eval_iter = DataLoader(dataset,
                       batch_size=args.eval_batch_size,
                       sampler=eval_sampler)

model = models[args.model_name]
model_params = { 
    "enc_hs": 128,
    "dec_hs": 64,
    "enc_hidden_size": 128,
    "dec_hidden_size": 64,
    "drop_prob": args.dropout,
    "device": device,
    "is_qat": args.qat,
    "offset": args.offset
    }
model = model(**model_params)
model.to(device)

optim = torch.optim.Adam(model.parameters(), lr=args.lr)

#############################################################################################
##  loading model
############################################################################################
if args.restart:
    checkpoints = torch.load(args.ckpt_path)
    model.load_state_dict(checkpoints['model_state_dict'])
    optim.load_state_dict(checkpoints['optim_state_dict'])
    best_score = checkpoints['score']

if args.fp8:
    model = torch.quantization.quantize_dynamic(model,
                                                {torch.nn.LSTM,
                                                 torch.nn.Linear},
                                                dtype=torch.quint8)


if args.qat:
    model.train()
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    model_fp32_fused = torch.quantization.fuse_modules(model,
                                                       [['encoder', 'decoder', 'full_connected']])

    model = torch.quantization.prepare_qat(model_fp32_fused)



#############################################################################################
##  train model
#############################################################################################
if 'depressed' in args.model_name:
    ModelTrainer = DepressModelTrainer
else:
    ModelTrainer = LSTMModelTrainer


trainer_params = {
    "epochs": args.epochs,
    "train_iter": train_iter,
    "dev_iter": dev_iter,
    "logger": logging,
    "batch_size": args.batch_size,
    "log_interval": args.log_interval,
    "eval_interval": args.eval_interval,
    "optim": optim,
    "save_path": save_path,
    "is_qat": args.qat
}

model_trainer = ModelTrainer(**trainer_params)

model_trainer.train(model, None)

#############################################################################################
##  evaluate model
#############################################################################################

model.eval()
tqdm_eval_iter = tqdm(eval_iter)
total_eval_score = 0
eval_start_time = time.time()
with torch.no_grad():
    for data in tqdm_eval_iter:
        _, eval_score = model(data)
        total_eval_score += eval_score

eval_avg_score = total_eval_score / len(tqdm_eval_iter)
eval_log_str  = f"| Final Eval | speed time: {time.time() - eval_start_time} |" \
                f"| average eval loss: {eval_avg_score} | "

logging('-' * 100 + "\n" + eval_log_str + "\n" + '-' * 100, print_=True)


#############################################################################################
##  save model
#############################################################################################

if args.qat:
    model = torch.quantization.convert(model)

save_params = {
    "model_state_dict": model.state_dict(),
    "optim_state_dict": optim.state_dict(),
    "score": eval_avg_score
}
torch.save(save_params,
           os.path.join(save_path, "LSTM.pt"))






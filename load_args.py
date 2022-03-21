import argparse



def load_args():
    parser = argparse.ArgumentParser(description="train model")

    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--epochs", type=int, default=1024, help="number of epochs for training")
    parser.add_argument("--seed", type=int, default=44, help="random seed")
    parser.add_argument("--dropout", type=float, default=0.75, help="setting dropout probability")
    parser.add_argument("--debug", action="store_true", help="open debug mode")
    parser.add_argument("--restart", action="store_true", help="restart model")
    parser.add_argument("--ckpt_path", type=str, default="/home/cy/workspace/npp/save/LSTM", help="path to load checkpoints")
    parser.add_argument("--cuda", action="store_false", help="use gpu")
    parser.add_argument("--log_interval", type=int, default=200, help="report interval")
    parser.add_argument("--eval_interval", type=int, default=8, help="the number of epochs to evaluation interval")
    parser.add_argument("--offset", type=int, default=6, help="the number of offset t predict. (17ms per t)")
    parser.add_argument("--batch_size", type=int, default=512, help="batch size for train")
    parser.add_argument("--eval_batch_size", type=int, default=1024, help="batch size for evaluation")
    parser.add_argument("--data_path", type=str, default="/home/cy/workspace/datasets/new_240hz_data", help="data")
    parser.add_argument("--load_dataset_path", type=str, default="", help="dataset")
    parser.add_argument("--proj_name", type=str, default="npp", help="data")
    parser.add_argument("--model_name", type=str, default="lstm", help="select model")
    parser.add_argument("--rank", type=str, default="0", help="gpu rank")
    parser.add_argument("--qat", action="store_true", help="qat")
    parser.add_argument("--fp8", action="store_true", help="float 8bit")
    args = parser.parse_args()

    return args

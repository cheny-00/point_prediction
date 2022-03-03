import argparse



def load_args():
    parser = argparse.ArgumentParser(description="train model")

    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--epochs", type=int, default=64, help="number of epochs for training")
    parser.add_argument("--dropout", type=float, default=0.75, help="setting dropout probability")
    parser.add_argument("--debug", action="store_true", help="open debug mode")
    parser.add_argument("--cuda", action="store_false", help="use gpu")
    parser.add_argument("--log_interval", type=int, default=200, help="report interval")
    parser.add_argument("--eval_interval", type=int, default=8, help="the number of epochs to evaluation interval")
    parser.add_argument("--offset", type=int, default=2, help="the number of offset t predict. (17ms per t)")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size for training")
    args = parser.parse_args()

    return args
import onnx
import numpy as np
import onnxruntime as ort

from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.data import LSTMDataset
onnx_path = ""
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)


offset = 6
batch_size = 1024
eval_data_path = ""
eval_dataset = LSTMDataset(data_path=eval_data_path,
                           offset=6)

eval_iter = DataLoader(eval_dataset,
                       batch_size=batch_size)


def ade(pred, target):
    dist = np.sum(np.sqrt(np.sum(np.pow(pred - target, 2), -1))) / len(pred)
    return dist

total_dist = 0
for data in eval_iter:
    inp = data['input'].numpy()
    label = data['label'].numpy()
    ort_sess = ort.InferenceSession(onnx_path)
    out = ort_sess.run(None, {'input_1': inp})
    total_dist += ade(out, label)
print(f"{offset * 4} ms {total_dist / len(eval_iter)} px")
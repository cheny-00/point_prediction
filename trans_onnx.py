import torch
from model.model import LSTMPointPredict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMPointPredict(enc_hidden_size=512,
                         dec_hidden_size=256,
                         drop_prob=0.75,
                         device=device)
ckpt_path = ""
ckpt = torch.load(ckpt_path)

model.load_state_dict(ckpt["model_state_dict"])
model.eval()

input_shape = (40, 3)
x = torch.randn(1, *input_shape)
export_onnx_file = "npp.onnx"
torch.onnx.export(model,
                  x,
                  export_onnx_file,
                  opset_version=10,
                  do_constant_folding=True,
                  input_names=["input_1"],
                  output_names=["dense"],
                  dynamic_axes={"input": {0: "batch_size"},
                                "output": {0: "batch_size"}})


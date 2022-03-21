import torch
from model.onnx_model import LSTMPointPredict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMPointPredict(enc_hidden_size=512,
                        dec_hidden_size=256,
                        drop_prob=0.75)
for i in [6,7,8]:
    ckpt_path = f"/home/cy/workspace/npp/models_4ms/npp_{i}.pt"
    ckpt = torch.load(ckpt_path)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    input_shape = (40, 3)
    x = torch.randn(1, *input_shape)
    export_onnx_file = f"npp_{i}.onnx"
    torch.onnx.export(model,
                        x,
                        export_onnx_file,
                        opset_version=14,
                        do_constant_folding=True,
                        input_names=["input_1"],
                        output_names=["dense"])


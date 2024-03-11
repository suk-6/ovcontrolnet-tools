import torch
import openvino as ov
from pathlib import Path

DEVICE = "CPU"
OPENPOSE_OV_PATH = Path("./models/openpose.xml")
CONTROLNET_OV_PATH = Path("./models/controlnet-pose.xml")
UNET_OV_PATH = Path("./models/unet_controlnet.xml")
TEXT_ENCODER_OV_PATH = Path("./models/text_encoder.xml")
VAE_DECODER_OV_PATH = Path("./models/vae_decoder.xml")

dtype_mapping = {
    torch.float32: ov.Type.f32,
    torch.float64: ov.Type.f64,
    torch.int32: ov.Type.i32,
    torch.int64: ov.Type.i64,
}

inputs = {
    "sample": torch.randn((2, 4, 64, 64)),
    "timestep": torch.tensor(1),
    "encoder_hidden_states": torch.randn((2, 77, 768)),
    "controlnet_cond": torch.randn((2, 3, 512, 512)),
}

input_info = [(name, ov.PartialShape(inp.shape)) for name, inp in inputs.items()]

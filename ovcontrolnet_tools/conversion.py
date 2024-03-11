import gc
import torch
import openvino as ov
from functools import partial

from ovcontrolnet_tools.utils import *
from ovcontrolnet_tools.options import *
from ovcontrolnet_tools.models import *


# Convert models to OpenVINO Intermediate representation (IR) format
# OpenPose conversion
def openposeConversion():
    if not OPENPOSE_OV_PATH.exists():
        with torch.no_grad():
            ov_model = ov.convert_model(
                pose_estimator.body_estimation.model,
                example_input=torch.zeros([1, 3, 184, 136]),
                input=[[1, 3, 184, 136]],
            )
            ov.save_model(ov_model, OPENPOSE_OV_PATH)
            del ov_model
            cleanup_torchscript_cache()
        print("OpenPose successfully converted to IR")
    else:
        print(f"OpenPose will be loaded from {OPENPOSE_OV_PATH}")

    ov_openpose = OpenPoseOVModel(core, OPENPOSE_OV_PATH, device=DEVICE)
    pose_estimator.body_estimation.model = ov_openpose


# ControlNet conversion
def controlnetConversion():
    global controlnet
    controlnet.eval()
    with torch.no_grad():
        down_block_res_samples, mid_block_res_sample = controlnet(
            **inputs, return_dict=False
        )

    if not CONTROLNET_OV_PATH.exists():
        with torch.no_grad():
            controlnet.forward = partial(controlnet.forward, return_dict=False)
            ov_model = ov.convert_model(
                controlnet, example_input=inputs, input=input_info
            )
            ov.save_model(ov_model, CONTROLNET_OV_PATH)
            del ov_model
            cleanup_torchscript_cache()
        print("ControlNet successfully converted to IR")
    else:
        print(f"ControlNet will be loaded from {CONTROLNET_OV_PATH}")

    del controlnet
    gc.collect()

    return down_block_res_samples, mid_block_res_sample


# Unet conversion
def unetConversion(down_block_res_samples, mid_block_res_sample):
    if not UNET_OV_PATH.exists():
        inputs.pop("controlnet_cond", None)
        inputs["down_block_additional_residuals"] = down_block_res_samples
        inputs["mid_block_additional_residual"] = mid_block_res_sample

        unet = UnetWrapper(pipe.unet)
        unet.eval()

        with torch.no_grad():
            ov_model = ov.convert_model(unet, example_input=inputs)

        flatten_inputs = flattenize_inputs(inputs.values())
        for input_data, input_tensor in zip(flatten_inputs, ov_model.inputs):
            input_tensor.get_node().set_partial_shape(ov.PartialShape(input_data.shape))
            input_tensor.get_node().set_element_type(dtype_mapping[input_data.dtype])
        ov_model.validate_nodes_and_infer_types()
        ov.save_model(ov_model, UNET_OV_PATH)
        del ov_model
        cleanup_torchscript_cache()
        del unet
        del pipe.unet
        gc.collect()
        print("Unet successfully converted to IR")
    else:
        del pipe.unet
        print(f"Unet will be loaded from {UNET_OV_PATH}")
    gc.collect()


# Text Encoder conversion
def convert_encoder(text_encoder: torch.nn.Module, ir_path: Path):
    if not ir_path.exists():
        input_ids = torch.ones((1, 77), dtype=torch.long)
        # switch model to inference mode
        text_encoder.eval()

        # disable gradients calculation for reducing memory consumption
        with torch.no_grad():
            ov_model = ov.convert_model(
                text_encoder,  # model instance
                example_input=input_ids,  # inputs for model tracing
                input=([1, 77],),
            )
            ov.save_model(ov_model, ir_path)
            del ov_model
        cleanup_torchscript_cache()
        print("Text Encoder successfully converted to IR")


def textEncoderConversion():
    if not TEXT_ENCODER_OV_PATH.exists():
        convert_encoder(pipe.text_encoder, TEXT_ENCODER_OV_PATH)
    else:
        print(f"Text encoder will be loaded from {TEXT_ENCODER_OV_PATH}")
    del pipe.text_encoder
    gc.collect()


def convert_vae_decoder(vae: torch.nn.Module, ir_path: Path):
    class VAEDecoderWrapper(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, latents):
            return self.vae.decode(latents)

    if not ir_path.exists():
        vae_decoder = VAEDecoderWrapper(vae)
        latents = torch.zeros((1, 4, 64, 64))

        vae_decoder.eval()
        with torch.no_grad():
            ov_model = ov.convert_model(
                vae_decoder,
                example_input=latents,
                input=[
                    (1, 4, 64, 64),
                ],
            )
            ov.save_model(ov_model, ir_path)
        del ov_model
        cleanup_torchscript_cache()
        print("VAE decoder successfully converted to IR")


def vaeDecoderConversion():
    if not VAE_DECODER_OV_PATH.exists():
        convert_vae_decoder(pipe.vae, VAE_DECODER_OV_PATH)
    else:
        print(f"VAE decoder will be loaded from {VAE_DECODER_OV_PATH}")


def IRConversion():
    openposeConversion()
    down_block_res_samples, mid_block_res_sample = controlnetConversion()
    unetConversion(down_block_res_samples, mid_block_res_sample)
    textEncoderConversion()
    vaeDecoderConversion()

import torch
import openvino as ov
from typing import Tuple
from collections import namedtuple
from controlnet_aux import OpenposeDetector
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel


class OpenPoseOVModel:
    """Helper wrapper for OpenPose model inference"""

    def __init__(self, core, model_path, device="AUTO"):
        self.core = core
        self.model = core.read_model(model_path)
        self.compiled_model = core.compile_model(self.model, device)

    def __call__(self, input_tensor: torch.Tensor):
        """
        inference step

        Parameters:
          input_tensor (torch.Tensor): tensor with prerpcessed input image
        Returns:
           predicted keypoints heatmaps
        """
        h, w = input_tensor.shape[2:]
        input_shape = self.model.input(0).shape
        if h != input_shape[2] or w != input_shape[3]:
            self.reshape_model(h, w)
        results = self.compiled_model(input_tensor)
        return torch.from_numpy(
            results[self.compiled_model.output(0)]
        ), torch.from_numpy(results[self.compiled_model.output(1)])

    def reshape_model(self, height: int, width: int):
        """
        helper method for reshaping model to fit input data

        Parameters:
          height (int): input tensor height
          width (int): input tensor width
        Returns:
          None
        """
        self.model.reshape({0: [1, 3, height, width]})
        self.compiled_model = self.core.compile_model(self.model)

    def parameters(self):
        Device = namedtuple("Device", ["device"])
        return [Device(torch.device("cpu"))]


core = ov.Core()

# Convert models to OpenVINO Intermediate representation (IR) format
pose_estimator = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch.float32
)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet
)


class UnetWrapper(torch.nn.Module):
    def __init__(
        self,
        unet,
        sample_dtype=torch.float32,
        timestep_dtype=torch.int64,
        encoder_hidden_states=torch.float32,
        down_block_additional_residuals=torch.float32,
        mid_block_additional_residual=torch.float32,
    ):
        super().__init__()
        self.unet = unet
        self.sample_dtype = sample_dtype
        self.timestep_dtype = timestep_dtype
        self.encoder_hidden_states_dtype = encoder_hidden_states
        self.down_block_additional_residuals_dtype = down_block_additional_residuals
        self.mid_block_additional_residual_dtype = mid_block_additional_residual

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        down_block_additional_residuals: Tuple[torch.Tensor],
        mid_block_additional_residual: torch.Tensor,
    ):
        sample.to(self.sample_dtype)
        timestep.to(self.timestep_dtype)
        encoder_hidden_states.to(self.encoder_hidden_states_dtype)
        down_block_additional_residuals = [
            res.to(self.down_block_additional_residuals_dtype)
            for res in down_block_additional_residuals
        ]
        mid_block_additional_residual.to(self.mid_block_additional_residual_dtype)
        return self.unet(
            sample,
            timestep,
            encoder_hidden_states,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual,
        )

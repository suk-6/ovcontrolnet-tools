from transformers import CLIPTokenizer
from diffusers import UniPCMultistepScheduler

from ovcontrolnet_tools.utils import *
from ovcontrolnet_tools.options import *
from ovcontrolnet_tools.models import *
from ovcontrolnet_tools.pipe import *
from ovcontrolnet_tools.conversion import IRConversion


def getOVPipe():
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    ov_pipe = OVContrlNetStableDiffusionPipeline(
        tokenizer,
        scheduler,
        core,
        CONTROLNET_OV_PATH,
        TEXT_ENCODER_OV_PATH,
        UNET_OV_PATH,
        VAE_DECODER_OV_PATH,
        device=DEVICE,
    )

    np.random.seed(42)

    return ov_pipe


if __name__ == "__main__":
    # Example usage
    from diffusers.utils import load_image

    original = load_image(
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/person.png"
    )
    IRConversion()
    pose = pose_estimator(original)

    prompt = "Dancing Darth Vader, best quality, extremely detailed"
    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

    ov_pipe = getOVPipe()
    result = ov_pipe(prompt, pose, 20, negative_prompt=negative_prompt)
    make_results(original, pose, result[0])

from ovcontrolnet_tools import *

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

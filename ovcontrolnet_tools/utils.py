import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def cleanup_torchscript_cache():
    """
    Helper for removing cached model representation
    """
    torch._C._jit_clear_class_registry()
    torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
    torch.jit._state._clear_class_state()


def flattenize_inputs(inputs):
    flatten_inputs = []
    for input_data in inputs:
        if input_data is None:
            continue
        if isinstance(input_data, (list, tuple)):
            flatten_inputs.extend(flattenize_inputs(input_data))
        else:
            flatten_inputs.append(input_data)
    return flatten_inputs


def make_results(
    orig_img: Image.Image, skeleton_img: Image.Image, result_img: Image.Image
):
    orig_title = "Original image"
    skeleton_title = "Pose"
    orig_img = orig_img.resize(result_img.size)
    im_w, im_h = orig_img.size
    is_horizontal = im_h <= im_w
    figsize = (20, 20)
    fig, axs = plt.subplots(
        3 if is_horizontal else 1,
        1 if is_horizontal else 3,
        figsize=figsize,
        sharex="all",
        sharey="all",
    )
    fig.patch.set_facecolor("white")
    list_axes = list(axs.flat)
    for a in list_axes:
        a.set_xticklabels([])
        a.set_yticklabels([])
        a.get_xaxis().set_visible(False)
        a.get_yaxis().set_visible(False)
        a.grid(False)
    list_axes[0].imshow(np.array(orig_img))
    list_axes[1].imshow(np.array(skeleton_img))
    list_axes[2].imshow(np.array(result_img))
    list_axes[0].set_title(orig_title, fontsize=15)
    list_axes[1].set_title(skeleton_title, fontsize=15)
    list_axes[2].set_title("Result", fontsize=15)
    fig.subplots_adjust(
        wspace=0.01 if is_horizontal else 0.00, hspace=0.01 if is_horizontal else 0.1
    )
    fig.tight_layout()
    fig.savefig("result.png", bbox_inches="tight")
    return fig

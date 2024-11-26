"""
mindspore 2.4.0
"""

import mindspore as ms


if __name__ == "__main__":
    # load model
    # ckpt = ms.load_checkpoint(
    #     "./results/20241112LF-qwen2-7B-Instruct/adapter_model.safetensors",
    #     format="safetensors",
    # )
    # print(ckpt.keys())

    # convert model
    ms.safetensors_to_ckpt(
        "./results/20241112LF-qwen2-7B-Instruct/adapter_model.safetensors",
        "./results/20241112LF-qwen2-7B-Instruct",
    )

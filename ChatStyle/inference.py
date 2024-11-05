import argparse
import mindspore as ms
from mindspore import context
from mindnlp.peft import PeftModel, PeftConfig
from mindnlp.transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    ChatGLM3Tokenizer,
    ChatGLM3ForConditionalGeneration,
)

import datasets

context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
context.set_context(device_id=1)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen2.5-3B",  # Qwen/Qwen2-7B-Instruct
    )
    parser.add_argument("--peft_type", type=str, default="LoRA")
    parser.add_argument("--task_type", type=str, default="CAUSAL_LM")
    parser.add_argument("--train_file", type=str, default="./XiYouJi/XiYouJi.json")
    parser.add_argument("--inf_max_length", type=int, default=128)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    return parser.parse_args()


def inference(args):
    peft_model_id = (
        f"{args.save_dir}/{args.model_name_or_path}_{args.peft_type}_{args.task_type}"
    )

    # config = PeftConfig.from_pretrained(peft_model_id)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).half()
    model = PeftModel.from_pretrained(model, peft_model_id)
    model.set_train(False)

    with ms._no_grad():
        while True:
            inputs = input("Q: ")
            if inputs in ("exit", "Exit", "quit", "Quit", "e", "q"):
                break
            inputs = (
                "【System】将下面的句子转换为文言文风格。【User】"
                + inputs
                + "【Assitant】"
            )  # + "【Assitant】"
            # inputs = "[System]: 将白话文转换成文言文。[User]: "+inputs+"[Assistant]: 行者道："
            message = tokenizer(inputs, return_tensors="ms")
            message["max_new_tokens"] = args.inf_max_length
            print(f"{message}")

            outputs = model.generate(**message)
            # outputs,_ = model.chat(tokenizer,query=dataset["validation"][text_column][i], max_length=128)
            # print(outputs)
            text_output = tokenizer.batch_decode(
                outputs.asnumpy(), skip_special_tokens=True
            )
            print(f"A: {text_output}")


if __name__ == "__main__":
    args = get_args()
    inference(args)

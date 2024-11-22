from typing import List, Dict, Tuple
import argparse
import mindspore as ms
from mindspore import context
from mindnlp.peft import PeftModel, PeftConfig
from mindnlp.transformers import AutoModelForCausalLM, AutoTokenizer

import datasets

context.set_context(device_target="GPU")
context.set_context(device_id=1)

prompt_system = "<|im_start|>system\n{}<|im_end|>\n"
prompt_user = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
prompt_assistant = "{}<|im_end|>\n"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="./ChatStyle/.mindnlp/model/Qwen/Qwen2-7B-Instruct",  # Qwen/Qwen2-7B-Instruct
    )
    parser.add_argument("--inf_max_length", type=int, default=128)
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="./ChatStyle/results/20241112LF-qwen2-7B-Instruct",
    )
    return parser.parse_args()


def init_socket():
    # TODO init Websocket server
    pass


def get_prompt(msgs: List[Dict]):
    text = ""
    for i in range(len(msgs)):
        if msgs[i]["role"] == "system":
            text += prompt_system.format(msgs[i]["content"])
        elif msgs[i]["role"] == "user":
            text += prompt_user.format(msgs[i]["content"])
        else:
            text += prompt_assistant.format(msgs[i]["content"])
    text += "行者道："

    return text


def get_RAG_prompt(msgs: List[Dict], query: str):
    # TODO get RAG prompt
    pass


def inference(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, ms_dtype=ms.float16
    )
    # model = PeftModel.from_pretrained(model, args.adapter_path)
    model.set_train(False)

    messages = [
        {"role": "system", "content": "假如你是《西游记》中的某个角色，请与我对话。"}
    ]
    with ms._no_grad():
        while True:
            inputs = input("Q: ")
            if inputs in ("exit", "Exit", "quit", "Quit", "e", "q"):
                break
            messages.append(
                {"role": "user", "content": inputs},
            )
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            text = get_prompt(messages)
            print((f"text: {text}", f"prompt: {prompt}"))
            model_inputs = tokenizer([prompt], return_tensors="ms")
            model_inputs["max_new_tokens"] = args.inf_max_length
            print(f"{model_inputs}")

            outputs = model.generate(**model_inputs)
            outputs = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(model_inputs["input_ids"], outputs)
            ]
            # print(outputs)
            text_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            print(f"A: {text_output}")

            messages.append({"role": "assistant", "content": text_output})


if __name__ == "__main__":
    args = get_args()
    inference(args)

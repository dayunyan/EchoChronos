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
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model = PeftModel.from_pretrained(model, peft_model_id)
    model.set_train(False)

    prompt_system = "<|im_start|>system\n{}<|im_end|>\n"
    prompt_user = "<|im_start|>{}\n<|im_end|>\n<|im_start|>"
    prompt_assistant = "{}\n<|im_end|>\n"
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
            # text = tokenizer.apply_chat_template(
            #     message,
            #     tokenize=False,
            #     truncation=True,
            #     return_tensors="ms",
            #     add_generation_prompt=True,
            # )
            text = ""
            for i in range(len(messages)):
                if messages[i]["role"] == "system":
                    text += prompt_system.format(messages[i]["content"])
                elif messages[i]["role"] == "user":
                    text += prompt_user.format(messages[i]["content"])
                else:
                    text += prompt_assistant.format(messages[i]["content"])
            text += "行者道："
            # inputs = "[System]: 将白话文转换成文言文。[User]: "+inputs+"[Assistant]: 行者道："
            model_inputs = tokenizer([text], return_tensors="ms")
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

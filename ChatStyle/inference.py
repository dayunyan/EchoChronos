import argparse
import mindspore as ms
from mindspore import context
from mindnlp.peft import PeftModel, PeftConfig
from mindnlp.transformers import (
    AutoModelForSeq2SeqLM,
    ChatGLM3Tokenizer,
    ChatGLM3ForConditionalGeneration,
)

import datasets

context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
context.set_context(device_id=1)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="ZhipuAI/chatglm3-6b")
    parser.add_argument("--peft_type", type=str, default="IA3")
    parser.add_argument("--task_type", type=str, default="CAUSAL_LM")
    parser.add_argument("--train_file", type=str, default="./XiYouJi/XiYouJi.json")
    parser.add_argument("--inf_max_length", type=int, default=64)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    return parser.parse_args()


@ms._no_grad()
def inference(args):
    peft_model_id = (
        f"{args.save_dir}/{args.model_name_or_path}_{args.peft_type}_{args.task_type}"
    )

    # config = PeftConfig.from_pretrained(peft_model_id)
    tokenizer = ChatGLM3Tokenizer.from_pretrained(args.model_name_or_path)
    model = ChatGLM3ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    model = PeftModel.from_pretrained(model, peft_model_id)
    model.set_train(False)

    while True:
        inputs = input("Q: ")
        input_ids = tokenizer(inputs + "行者道：", return_tensors="ms")
        # print(f"Q: {inputs}")

        outputs = model.generate(
            input_ids=input_ids["input_ids"], max_length=args.inf_max_length
        )
        # outputs,_ = model.chat(tokenizer,query=dataset["validation"][text_column][i], max_length=128)
        # print(outputs)
        text_output = tokenizer.batch_decode(
            outputs.asnumpy(), skip_special_tokens=True
        )
        print(f"A: {text_output}")

        if inputs in ("exit", "Exit", "quit", "Quit", "e", "q"):
            break


if __name__ == "__main__":
    args = get_args()
    inference(args)

from typing import List, Dict, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, peft_model

from managers.constants import *
from managers.rag import RAGManager


def get_model_kwargs(kwargs):
    if "torch_dtype" in kwargs:
        kwargs["torch_dtype"] = getattr(torch, kwargs["torch_dtype"])
    return kwargs


class ModelManager:
    def __init__(
        self,
        model_config: Dict,
        rag_config: Dict,
        # model_name_or_path: str,
        # inf_max_length: int,
        # adapter_path: str,
        # has_rag: bool,
        # **rag_args,
    ):
        self.prompt_system = "<|im_start|>system\n{}<|im_end|>\n"
        self.prompt_user = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        self.prompt_assistant = "{}<|im_end|>\n"

        self.model_config = model_config
        self.model_name_or_path = model_config.get(
            "model_name_or_path",
            model_config.get("model_path", model_config.get("model_name", None)),
        )
        # self.max_new_tokens = model_config.get("max_new_tokens", 1024)
        self.generate_kwargs = model_config.get("generate_kwargs", {})
        self.adapter_path = model_config.get("adapter_path", None)
        self.has_rag = model_config.get("has_rag", False)

        self.init_model()
        if self.has_rag:
            self.rag_mannager = RAGManager(rag_config)

    def init_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            **get_model_kwargs(self.model_config.get("model_kwargs", None)),
        )
        # lora_config = LoraConfig.from_pretrained(self.adapter_path)
        # self.model = get_peft_model(self.model, lora_config)
        self.model.load_adapter(self.adapter_path, device_map="auto")
        self.model.eval()

    def get_prompt(
        self,
        msgs: List[Dict],
        source: str = "西游记",
        character: str = "孙悟空",
        **kwargs,
    ):
        has_rag = kwargs.get("k", 0) and self.has_rag
        text = ""
        if has_rag:
            retrieved_info = self.rag_mannager.get_RAG_prompt(
                source=source,
                character=character,
                query=msgs[-1]["content"],
                **kwargs,
            )
        for i in range(len(msgs)):
            if msgs[i]["role"] == "system":
                text += self.prompt_system.format(msgs[i]["content"])
            elif msgs[i]["role"] == "assistant":
                text += self.prompt_assistant.format(msgs[i]["content"])
            else:
                user_input = ""
                if i == 1:
                    user_input += (
                        """假如你是<{book}>中的{role}，请与我对话。\n""".format(
                            book=source,
                            role=character,
                        )
                    )
                if has_rag and i == len(msgs) - 1:
                    user_input += """{retrieved_info}\n
                    参考以上信息，请与我对话。\n
                    {query}""".format(
                        retrieved_info=retrieved_info,
                        query=msgs[i]["content"],
                    )
                else:
                    user_input += """{query}""".format(
                        query=msgs[i]["content"],
                    )
                text += self.prompt_user.format(user_input)
        text += f"{CHARACTER_DICT[source][character]}道："

        return text

    @torch.inference_mode()
    def infer(
        self,
        messages: List[Dict],
        source: str = "西游记",
        character: str = "孙悟空",
        **kwargs,
    ):
        text = self.get_prompt(messages, source=source, character=character, **kwargs)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        # model_inputs["max_new_tokens"] = self.max_new_tokens
        # print(f"{model_inputs}")
        for gkw in self.generate_kwargs:
            self.generate_kwargs[gkw] = kwargs.get(gkw, self.generate_kwargs[gkw])
        outputs = self.model.generate(**model_inputs, **self.generate_kwargs)
        outputs = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs["input_ids"], outputs)
        ]
        # print(outputs)
        text_output = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        print(f"A: {text_output.strip('“”')}")

        messages.append({"role": "assistant", "content": text_output})

        return text_output.strip("“”")

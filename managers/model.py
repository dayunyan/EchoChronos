from typing import List, Dict, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, peft_model

from managers.constants import *
from managers.rag import RAGManager


class ModelManager:
    def __init__(
        self,
        model_name_or_path: str,
        inf_max_length: int,
        adapter_path: str,
        has_rag: bool,
        **rag_args,
    ):
        self.prompt_system = "<|im_start|>system\n{}<|im_end|>\n"
        self.prompt_user = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        self.prompt_assistant = "{}<|im_end|>\n"

        self.model_name_or_path = model_name_or_path
        self.inf_max_length = inf_max_length
        self.adapter_path = adapter_path
        self.has_rag = has_rag

        self.init_model()
        if self.has_rag:
            self.rag_mannager = RAGManager(**rag_args)

    def init_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
        lora_config = LoraConfig.from_pretrained(self.adapter_path)
        self.model = get_peft_model(model, lora_config)
        self.model.eval()

    def get_prompt(
        self, msgs: List[Dict], source: str = "西游记", role: str = "孙悟空"
    ):
        text = ""
        for i in range(len(msgs)):
            if msgs[i]["role"] == "system":
                text += self.prompt_system.format(msgs[i]["content"])
            elif msgs[i]["role"] == "user":
                retrieved_info = (
                    self.rag_mannager.get_RAG_prompt(source, role, msgs[i]["content"])
                    if self.has_rag
                    else ""
                )
                if i == 2:
                    user_input = """假如你是<{source}>中的{role}，请与我对话。我知道的有： \n
                    {retrieved_info}\n
                    请你回答这个问题： {query}""".format(
                        source=source,
                        role=role,
                        retrieved_info=retrieved_info,
                        query=msgs[i]["content"],
                    )
                else:
                    user_input = """我知道的有： \n
                    {retrieved_info}\n
                    请你回答这个问题： {query}""".format(
                        source=source,
                        role=role,
                        retrieved_info=retrieved_info,
                        query=msgs[i]["content"],
                    )
                text += self.prompt_user.format(user_input)
            else:
                text += self.prompt_assistant.format(msgs[i]["content"])
        text += f"{ROLE_DICT[source][role]}道："

        return text

    @torch.inference_mode()
    def infer(self, messages: List[Dict], source: str = "西游记", role: str = "孙悟空"):
        text = self.get_prompt(messages, source=source, role=role)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        model_inputs["max_new_tokens"] = self.inf_max_length
        # print(f"{model_inputs}")

        outputs = self.model.generate(**model_inputs)
        outputs = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs["input_ids"], outputs)
        ]
        # print(outputs)
        text_output = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        print(f"A: {text_output.strip('“”')}")

        messages.append({"role": "assistant", "content": text_output})

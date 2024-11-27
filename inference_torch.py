import json
from typing import List, Dict, Tuple
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, peft_model

import requests
import gradio as gr

ROLE_DICT = {
    "西游记": {
        "孙悟空": "悟空",
        "唐僧": "唐僧",
        "猪八戒": "八戒",
        "沙僧": "沙僧",
    },
    "三国演义": {
        "刘备": "玄德",
        "关羽": "云长",
        "张飞": "翼德",
        "曹操": "曹操",
        "诸葛亮": "孔明",
    },
    "水浒传": {
        "宋江": "宋公明",
        "卢俊义": "玉麒麟",
        "吴用": "智多星",
        "林冲": "豹子头",
    },
    "红楼梦": {
        "贾宝玉": "宝玉",
        "林黛玉": "黛玉",
        "薛宝钗": "宝钗",
        "王熙凤": "凤姐",
    },
}

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
    parser.add_argument(
        "--isTerminal",
        action="store_true",
        help="Whether to use terminal for inference",
    )
    parser.add_argument(
        "--isWebsocket",
        action="store_true",
        help="Whether to use websocket for inference",
    )
    return parser.parse_args()


# format RAG retrieval results
def format_docs(docs, wiki_docs=None):
    ans = "从古籍中检索到的信息如下：\n\n"
    for id, doc in enumerate(docs):
        ans += f"{id+1}. {doc.page_content}\n\n"
    if wiki_docs is not None:
        ans += "从维基百科中检索到的信息如下：\n\n"
        ans += f'{len(docs)+1}. {wiki_docs[0].metadata["summary"]}\n\n'
    # print(f'检索到的信息有：{ans}')
    return ans


def get_RAG_prompt(book: str = "西游记", role: str = "孙悟空", query: str = ""):
    if query is None and len(query) == 0:
        return None

    # TODO: update when env changed
    base_rag_path = "/root/work_dir/huawei-ict-2024/RAG/rain-rag"
    # TODO: change config info if necessary, such as embedding model PATH, retriever PATH, etc.

    import sys

    sys.path.append(base_rag_path)
    from config.ConfigLoader import ConfigLoader

    config = ConfigLoader(base_rag_path + "/config/config.yaml")

    # import logging
    # log_level = config.get("global.log_level", "INFO")
    # logging.basicConfig(level = log_level,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # logger = logging.getLogger(__name__)

    from retrievers.RetrieverCreator import RetrieverCreator
    from embeddings.TorchBgeEmbeddings import EmbeddingModelCreator

    # logger.info("Creating embedding model...")

    # TODO: add embedding_path
    embedding_path = ""
    embedding_creator = EmbeddingModelCreator(config, embedding_path)
    embedding_model = embedding_creator.create_embedding_model()

    # logger.info("Creating retriever...")

    vecDB_path = base_rag_path + config.get("vector_db.index_path")
    retriever = RetrieverCreator(
        config, embedding_model, vecDB_path, collection_name="four_famous"
    ).create_retriever()

    retrieved_docs = retriever.invoke(query)
    retrieved_info = format_docs(retrieved_docs, None)

    # from langchain.prompts import PromptTemplate
    # template = """假如你是<{book}>中的{role}，请与我对话。我知道的有： \n
    #     {retrieved_info}\n
    #     请你回答这个问题： {query}。\n
    #     {spec_role}道："""

    # input_dict = {"book": book, "role": role, "retrieved_info": retrieved_info, "query": query, "spec_role": ROLE_DICT[book][role]}
    # input = template.format(**input_dict)

    return retrieved_info


def get_prompt(
    msgs: List[Dict], book: str = "西游记", role: str = "孙悟空", has_RAG=True
):
    text = ""
    for i in range(len(msgs)):
        if msgs[i]["role"] == "system":
            text += prompt_system.format(msgs[i]["content"])
        elif msgs[i]["role"] == "user":
            retrieved_info = (
                get_RAG_prompt(book, role, msgs[i]["content"]) if has_RAG else ""
            )
            if i == 2:
                user_input = """假如你是<{book}>中的{role}，请与我对话。我知道的有： \n
                {retrieved_info}\n
                请你回答这个问题： {query}""".format(
                    book=book,
                    role=role,
                    retrieved_info=retrieved_info,
                    query=msgs[i]["content"],
                )
            else:
                user_input = """我知道的有： \n
                {retrieved_info}\n
                请你回答这个问题： {query}""".format(
                    book=book,
                    role=role,
                    retrieved_info=retrieved_info,
                    query=msgs[i]["content"],
                )
            text += prompt_user.format(user_input)
        else:
            text += prompt_assistant.format(msgs[i]["content"])
    text += f"{ROLE_DICT[book][role]}道："

    return text


def processTTS(
    character: str = "sunwukong",
    # book: str = "西游记",
    text: str = "我是孙悟空",
):
    # 设置服务器地址和端口
    server_url = "http://172.24.244.174:5000/tts"  # 服务器地址根据实际情况修改
    prompt_language = "中文"  # 参考文本的语言
    text_language = "中文"  # 目标文本的语言
    how_to_cut = "不切"  # 文本切分方式
    top_k = 20  # Top-K 参数
    top_p = 0.6  # Top-P 参数
    temperature = 0.6  # 温度参数
    ref_free = False  # 是否使用参考音频

    # 准备POST请求的payload
    data = {
        "character": character,
        # 'prompt_text': prompt_text,
        "prompt_language": prompt_language,
        "text": text,
        "text_language": text_language,
        "how_to_cut": how_to_cut,
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "ref_free": ref_free,
    }

    # 发送POST请求
    response = requests.post(
        server_url,
        # files=files,
        data=data,
    )

    # 处理返回的音频文件
    if response.status_code == 200:
        # 保存返回的音频到文件
        with open("output_audio.wav", "wb") as f:
            f.write(response.content)
        print("Audio saved as output_audio.wav")
    else:
        print(f"Error: {response.status_code}, {response.text}")

    # 关闭文件
    # files['ref_wav'].close()


@torch.inference_mode()
def run(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    lora_config = LoraConfig.from_pretrained(args.adapter_path)
    model = get_peft_model(model, lora_config)
    model.eval()

    messages = [
        {
            "role": "system",
            "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        },
    ]

    def infer(user_inputs):
        messages.append(
            {"role": "user", "content": user_inputs},
        )
        # prompt = tokenizer.apply_chat_template(
        #     messages,
        #     tokenize=False,
        #     add_generation_prompt=True,
        # )
        text = get_prompt(messages, book="红楼梦", role="林黛玉")  # TODO get RAG prompt

        # print((f"text: {text}", f"prompt: {prompt}"))
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        model_inputs["max_new_tokens"] = args.inf_max_length
        # print(f"{model_inputs}")

        outputs = model.generate(**model_inputs)
        outputs = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs["input_ids"], outputs)
        ]
        # print(outputs)
        text_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        print(f"A: {text_output.strip('“”')}")

        messages.append({"role": "assistant", "content": text_output})
        return text_output

    if args.isTerminal:
        while True:
            inputs = input("Q: ")
            outputs = infer(inputs)
            processTTS(character="lindaiyu", text=outputs)
    elif args.isWebsocket:

        def init_socket():
            import asyncio
            import websockets

            async def echo(websocket):
                async for message in websocket:
                    client_msg = json.loads(message)
                    print(f"Received message from client: {client_msg}")
                    response = {
                        "status": "success",
                        "echo": infer(client_msg["message"]),
                    }
                    processTTS(character="lindaiyu", text=response["echo"])
                    await websocket.send(json.dumps(response))

            async def main():
                async with websockets.serve(
                    echo, "0.0.0.0", 6006, ping_interval=None
                ):  # 0.0.0.0表示监听所有可用的网络接口，6006表示监听的端口号，需要根据防火墙规则确定
                    print("WebSocket server started on ws://172.16.185.158:6006")
                    await asyncio.Future()  # 运行直到被取消

            asyncio.run(main())

        init_socket()


if __name__ == "__main__":
    args = get_args()
    run(args)

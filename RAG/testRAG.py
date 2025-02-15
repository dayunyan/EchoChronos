from typing import List, Dict, Tuple
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, peft_model


ROLE_DICT = {
    "西游记": {
        "孙悟空": "行者",
        "唐僧": "唐僧",
        "猪八戒": "猪八戒",
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
    return parser.parse_args()


# format RAG retrieval results
def format_docs(docs, wiki_docs=None):
        ans = "从古籍中检索到的信息如下：\n\n"
        for id, doc in enumerate(docs):
            ans += f'{id+1}. {doc.page_content}\n\n'
        if wiki_docs is not None:
            ans += "从维基百科中检索到的信息如下：\n\n"
            ans += f'{len(docs)+1}. {wiki_docs[0].metadata["summary"]}\n\n'
        # print(f'检索到的信息有：{ans}')
        return ans


def get_RAG_prompt(msgs: List[Dict], book: str = "西游记", role: str = "孙悟空", query: str="", model: AutoModelForCausalLM=None, tokenizer: AutoTokenizer=None):
    if query is None and len(query) == 0:
        return None
    
    assert model is not None and tokenizer is not None, "model&tokenizer are required but were not provided"

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

    vecDB_path = base_rag_path + "/" + config.get('vector_db.index_path')
    retriever = RetrieverCreator(config, embedding_model, vecDB_path, collection_name="four_famous").create_retriever()

    template_retrieved = "在{book}中, {query}"
    retrieved_dict = {"book": book, "query": query}
    retrieved_docs = retriever.invoke(template_retrieved.format(**retrieved_dict))
    retrieved_info = format_docs(retrieved_docs, None)

    template = """假如你是<{book}>中的{role}，请与我对话。下面是已知信息： \n
        {retrieved_info}\n
        请你根据这些信息回答这个问题：{query}。\n
        {spec_role}道：“"""
    
    input_dict = {"book": book, "role": role, "retrieved_info": retrieved_info, "query": query, "spec_role": ROLE_DICT[book][role]}
    input = template.format(**input_dict)

    # print(input[:256])
    # logger.info("Creating model input...")
    model_inputs = tokenizer([input], return_tensors="pt").to(model.device)
    model_inputs["max_new_tokens"] = args.inf_max_length

    # logger.info("Creating RAG prompt...")
    outputs = model.generate(**model_inputs)
    outputs = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs["input_ids"], outputs)
    ]
    text_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    return text_output

# test passed
def test_RAG_prompt(args):
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
    with torch.no_grad():
        inputs = "周瑜是谁？"
        messages.append(
            {"role": "user", "content": inputs},
        )
        # def get_RAG_prompt(msgs: List[Dict], book: str = "西游记", role: str = "孙悟空", query: str="", model: AutoModelForCausalLM=None, tokenizer: AutoTokenizer=None):

        text_output = get_RAG_prompt(messages, book="三国演义", role="诸葛亮", query=inputs, model=model, tokenizer=tokenizer)
        print(f"A: {text_output.strip('”')}")


if __name__ == "__main__":
    args = get_args()
    test_RAG_prompt(args)

import logging
import os
from typing import List, Dict, Tuple
import mindspore as ms
from mindnlp.peft import PeftModel, PeftConfig, LoraConfig, get_peft_model
from mindnlp.transformers import AutoModelForCausalLM, AutoTokenizer
import streamlit as st
import requests
import base64

from RAG import ConfigLoader, EmbeddingModelCreator, RetrieverCreator

from llm.TorchModel_lora import Torch_Lora_LLM

from utils.yamlparam import YAMLParamHandler


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # 指定显卡

yaml_path = "./examples/infer_qwen2_lora_fp32_ms.yaml"
yaml_data = YAMLParamHandler(yaml_path).get_yaml_params()
rag_config = yaml_data.get("rag_config", {})


@st.cache_resource
def load_config(_config):
    return ConfigLoader(_config)


@st.cache_resource
def load_embedding(_config):
    embedding_creator = EmbeddingModelCreator(_config)
    embedding_model = embedding_creator.create_embedding_model()
    return embedding_model


# @st.cache_resource
def load_vecDB(_config, embedding_model):
    retriever_creator = RetrieverCreator(_config, embedding_model)
    return retriever_creator


@st.cache_resource
def load_Qwen2_7b_llm(_config):
    _config = _config["model"]
    model_name_or_path = _config.get("model_path", "model_name")
    model_kwargs = _config.get("model_kwargs", {})
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    adapter_path = _config.get("adapter_path", "adapter_name")
    lora_config = LoraConfig.from_pretrained(adapter_path)
    model = get_peft_model(model, lora_config)
    model.eval()
    return tokenizer, model


def format_docs(docs, wiki_docs=None):
    ans = "从古籍中检索到的信息如下：\n\n"
    for id, doc in enumerate(docs):
        ans += f"{id+1}. {doc.page_content}\n\n"
    if wiki_docs is not None:
        ans += "从维基百科中检索到的信息如下：\n\n"
        ans += f'{len(docs)+1}. {wiki_docs[0].metadata["summary"]}\n\n'
    # print(f'检索到的信息有：{ans}')
    return ans


def get_prompt(
    msgs: List[Dict],
    source: str = "西游记",
    role: str = "孙悟空",
    has_RAG=True,
    rag_info: str = "",
):
    text = ""
    for i in range(len(msgs)):
        if msgs[i]["role"] == "system":
            text += prompt_system.format(msgs[i]["content"])
        elif msgs[i]["role"] == "user":
            retrieved_info = rag_info if has_RAG else ""
            if i == 1:
                if i == len(msgs) - 1:
                    user_input = """假如你是<{source}>中的{role}，请与我对话。\n
                    {retrieved_info}\n
                    参考以上信息，与我对话。\n
                    {query}""".format(
                        source=source,
                        role=role,
                        retrieved_info=retrieved_info,
                        query=msgs[i]["content"],
                    )
                else:
                    user_input = """假如你是<{source}>中的{role}，请与我对话。\n
                    {query}""".format(
                        source=source,
                        role=role,
                        query=msgs[i]["content"],
                    )
            else:
                if i == len(msgs) - 1:
                    user_input = """{retrieved_info}\n
                    参考以上信息，与我对话。\n
                    {query}""".format(
                        retrieved_info=retrieved_info,
                        query=msgs[i]["content"],
                    )
                else:
                    user_input = """{query}""".format(
                        query=msgs[i]["content"],
                    )

            text += prompt_user.format(user_input)
        else:
            text += prompt_assistant.format(msgs[i]["content"])
    text += f"{ROLE_DICT[source][role]}道："

    return text


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
        "曹操": "孟德",
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

# 初始化历史对话记录
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        },
    ]

# if 'retrieve' not in st.session_state:
#     st.session_state.retrieve = None

# def set_retrieve(info):
#     st.session_state.retrieve = info

if "audio" not in st.session_state:
    st.session_state.audio = []


def append_audio(audio):
    st.session_state.audio.append(audio)


# 设置服务器地址和端口
server_url = (
    yaml_data["runner"].get("tts_server")
    if yaml_data["runner"].get("has_tts", False)
    else None
)


# 定义生成音频的函数
def generate_audio(character, text):
    # 设置要传递的参数
    prompt_language = "中文"  # 参考文本的语言
    text_language = "中文"  # 目标文本的语言
    how_to_cut = "按中文句号。切"  # 文本切分方式
    top_k = 20  # Top-K 参数
    top_p = 0.8  # Top-P 参数
    temperature = 0.6  # 温度参数
    ref_free = False  # 是否使用参考音频

    # 准备POST请求的payload
    data = {
        "character": character,
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
    response = requests.post(server_url, data=data)

    if response.status_code == 200:
        # 返回的音频文件内容
        audio_content = response.content
        return audio_content
    else:
        return f"Error: {response.status_code}, {response.text}"


# 设置页面标题
st.set_page_config(page_title="EchoChronos", layout="wide", page_icon="🦜")

st.sidebar.title("Configuration")
temperature = st.sidebar.slider("Temperature:", 0.01, 1.0, 1.0)
top_k = st.sidebar.slider("Top K:", 1, 50, 10)
top_p = st.sidebar.slider("Top P:", 0.01, 1.0, 0.7)
numk = st.sidebar.slider("Number of retrieved documents:", 0, 10, 3)
max_new_tokens = st.sidebar.slider("Max generation tokens:", 1, 2048, 512)


# 自定义CSS样式，设置聊天气泡的外观
st.markdown(
    """
    <style>
        .chat-bubble {
            display: inline-block;
            padding: 10px;
            border-radius: 20px;
            margin: 5px 0;
            max-width: 60%;
            font-size: 22px;
        }
        .user {
            background-color: #DCF8C6;
            text-align: left;
        }
        .assistant {
            background-color: #FFA500;
            text-align: left;
        }
        .chat-container {
            display: flex;
            flex-direction: column;
            align-items: flex-start;
            justify-content: flex-start;
        }
        .audio-container {
            margin-left: 0px;  /* 控制音频控件的缩进位置 */
            margin-top: 5px;
        }
       
    </style>
""",
    unsafe_allow_html=True,
)

# 构建页面
st.title("EchoChronos🦜")
st.write(
    f"<span style='font-size: 22px; color: #FF6347;'>对话之前记得先选择对话的角色呦❤️~~~</span>",
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2)
with col1:
    source = st.selectbox(
        "选择书籍", ["红楼梦", "西游记", "三国演义", "水浒传"], key="source"
    )
with col2:
    character = st.selectbox(
        "选择角色",
        ["林黛玉", "贾宝玉", "孙悟空", "猪八戒", "诸葛亮", "曹操", "林冲", "鲁智深"],
        key="char",
    )

# 用户输入框
# query = st.text_area("请输入你的问题：", height=100)

# LangChain 配置
template = """请仅根据以下信息回答，不要添加任何额外的假设或知识: \n
    {retrieved_info}\n
    请回答以下问题: {query}\n"""

config = load_config(rag_config)
tokenizer, model = load_Qwen2_7b_llm(yaml_data)
embedding = load_embedding(config)

vecDB = load_vecDB(config, embedding)

# prompt = PromptTemplate(template=template, input_variables=["query", "retrieved_info"])

log_level = yaml_data["global"].get("log_level", "INFO")
logging.basicConfig(
    level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# 显示对话
def display_messages():
    idx = 0
    for message in st.session_state.messages:
        # 用户消息
        if message["role"] == "user":
            col1, col2 = st.columns([1, 15])
            with col1:
                st.markdown(
                    f"<p style='font-size: 36px; color: gray;'> {'🤓'}</p>",
                    unsafe_allow_html=True,
                )  # 显示表情
            with col2:
                st.markdown(
                    f"<div class='chat-bubble user'>{message['content']}</div>",
                    unsafe_allow_html=True,
                )

        # 大模型回答
        elif message["role"] == "assistant":
            col1, col2 = st.columns([1, 15])
            with col1:
                st.markdown(
                    f"<p style='font-size: 36px; color: gray;'> {'🤖'}</p>",
                    unsafe_allow_html=True,
                )  # 显示表情
            with col2:
                st.markdown(
                    f"<div class='chat-bubble assistant'>{message['content']}</div>",
                    unsafe_allow_html=True,
                )

            col1, col2 = st.columns([1, 15])
            with col1:
                st.markdown(
                    f"<p style='font-size: 36px; color: gray;'> {'🤖'}</p>",
                    unsafe_allow_html=True,
                )  # 显示表情
            with col2:
                # 将音频字节数据编码为base64
                audio_base64 = base64.b64encode(st.session_state.audio[idx]).decode(
                    "utf-8"
                )
                audio_url = f"data:audio/wav;base64,{audio_base64}"
                st.markdown(
                    f'<div class="audio-container"><audio controls><source src="{audio_url}" type="audio/wav"></audio></div>',
                    unsafe_allow_html=True,
                )
                idx += 1


# 获取用户输入并生成回复
def user_input():
    query = st.text_input("请输入您的问题：", "")
    col1, col2 = st.columns([0.1, 1])
    with col1:
        send = st.button("发送", key="send")
    with col2:
        # clean = st.button("重新开始对话", on_click=restart, args=[], key="clean")
        clean = st.button("重新开始")

    if clean:
        st.session_state.messages.clear()
        st.session_state.messages = [
            {
                "role": "system",
                "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
            },
        ]
        st.session_state.audio.clear()

        st.rerun()

    if send:
        with st.spinner("正在生成回答..."):
            st.session_state.messages.append({"role": "user", "content": query})

            template_retrieved = "在{source}中, 针对{character}这个角色的提问：{query}"
            retrieved_dict = {"source": source, "character": character, "query": query}

            search_type = config.get(
                "langchain_modules.retrievers.vector_retriever.retrieval_type",
                "similarity",
            )
            search_kwargs = config.get(
                "langchain_modules.retrievers.vector_retriever.search_kwargs", {}
            )
            search_kwargs["k"] = numk

            retriever = vecDB.vector_store.as_retriever(
                search_type=search_type, search_kwargs=search_kwargs
            )

            retrieved_docs = retriever.invoke(
                template_retrieved.format(**retrieved_dict)
            )

            formatted_docs = format_docs(retrieved_docs)

            # logger.info(formatted_docs)

            inputs = get_prompt(
                st.session_state.messages,
                source=source,
                role=character,
                has_RAG=True,
                rag_info=formatted_docs,
            )

            # rag_res = llm(input)
            model_inputs = tokenizer([inputs], return_tensors="ms")
            generate_kwargs = yaml_data["model"].get("generate_kwargs", {})
            generated_ids = model.generate(
                input_ids=model_inputs.input_ids,
                **generate_kwargs,
            )
            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            rag_res = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            st.session_state.messages.append(
                {"role": "assistant", "content": rag_res.strip(" ").strip("“”")}
            )

            char_dict = {
                "林黛玉": "lindaiyu",
                "贾宝玉": "jiabaoyu",
                "诸葛亮": "zhugeliang",
                "曹操": "caocao",
                "孙悟空": "sunwukong",
                "猪八戒": "zhubajie",
                "林冲": "linchong",
                "鲁智深": "luzhishen",
            }
            # 生成音频
            audio_content = generate_audio(char_dict[character], rag_res)

            append_audio(audio_content)

            st.rerun()


# 调用显示函数
display_messages()
user_input()

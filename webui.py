import streamlit as st
import sys

# from utils.argparser import get_args, check_args
from utils.yamlparam import YAMLParamHandler
from managers.runner import RunnerManager


# 设置页面标题
st.set_page_config(page_title="EchoChronos", layout="wide", page_icon="🦜")

st.sidebar.title("Configuration")
temperature = st.sidebar.slider("Temperature:", 0.01, 1.0, 1.0)
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
    book = st.selectbox(
        "选择书籍", ["红楼梦", "西游记", "三国演义", "水浒传"], key="book"
    )
with col2:
    character = st.selectbox(
        "选择角色",
        ["林黛玉", "贾宝玉", "孙悟空", "猪八戒", "诸葛亮", "曹操", "林冲", "鲁智深"],
        key="char",
    )


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

            template_retrieved = "在{book}中, 针对{character}这个角色的提问：{query}"
            retrieved_dict = {"book": book, "character": character, "query": query}

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

            input = get_prompt(
                st.session_state.messages,
                book=book,
                role=character,
                has_RAG=True,
                rag_info=formatted_docs,
            )

            # rag_res = llm(input)
            model_inputs = llm.tokenizer([input], return_tensors="pt").to("cuda")
            generated_ids = llm.model.generate(
                model_inputs.input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            rag_res = llm.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

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

import base64
import json
from typing import List, Dict, Tuple
import asyncio

from managers.constants import *
from managers.model import ModelManager
from managers.tts import TTSManager
from managers.connect import WebSocketManager


class RunnerManager:
    def __init__(
        self,
        runner_config: Dict,
        model_config: Dict,
        rag_config: Dict,
    ):
        """_Initialize the RunnerManager_

        Args:
            runner_config (Dict): _runner_config_
                mode (str): the mode of the runner, optional["train", "inference"]
                source (str): the source from which the character comes, optional["西游记", "红楼梦", "三国演义", "水浒传"]
                character (str): the character, e.g. "孙悟空", "贾宝玉", "刘备", "宋江".
                isTerminal (bool): whether to run in terminal mode
                isWebsocket (bool): whether to run in websocket mode
                isWebUI (bool): whether to run in webUI mode
                host (str): the host of the websocket
                port (int): the port of the websocket
                has_tts (bool): whether to use TTS
                tts_server (str): the server of the TTS
            model_config (Dict): _model_config_
                model_name (str): the name of the model
                model_path (str): the path of the model
                max_new_tokens (int): Maximum length of the new tokens
                adapter_name (str): The name of the adapter
                adapter_path (str): The path of the adapter
                has_rag (bool): whether to use RAG
            rag_config (Dict): _description_
        """
        self.model_config = model_config
        self.rag_config = rag_config
        self.mode = runner_config.get("mode", "inference")
        self.source = runner_config.get("source", "西游记")
        self.character = runner_config.get("character", "孙悟空")
        self.messages = [
            {
                "role": "system",
                "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
            },
        ]
        self.isTerminal = runner_config.get("isTerminal", True)
        self.isWebsocket = runner_config.get("isWebsocket", False)
        self.isWebUI = runner_config.get("isWebUI", False)
        self.host = runner_config.get("host", "0.0.0.0")
        self.port = runner_config.get("port", 8080)

        # self.model_manager = ModelManager(
        #     model_config,
        #     rag_config,
        # )
        self.has_tts = runner_config.get("has_tts", False)
        if self.has_tts:
            self.tts_manager = TTSManager(
                runner_config.get("tts_server", "http://localhost:5000")
            )

    def init_model(self):
        self.model_manager = ModelManager(
            self.model_config,
            self.rag_config,
        )

    def run_terminal(self):
        self.init_model()
        while True:
            user_input = input("User: ")
            if user_input.lower() in ["e", "q", "exit", "quit"]:
                break
            self.messages.append(
                {
                    "role": "user",
                    "content": user_input,
                }
            )
            self.model_manager.infer(self.messages, self.source, self.character, k=3)
            if self.has_tts:
                self.tts_manager.process(
                    self.messages[-1]["role"], self.messages[-1]["content"]
                )

    def _run_websocket(self, msg):
        client_msg = json.loads(msg)
        if client_msg["role"] == "user":
            self.messages.append(
                {
                    "role": "user",
                    "content": client_msg["message"],
                }
            )
            self.model_manager.infer(self.messages, self.source, self.character)
            if self.has_tts:
                self.tts_manager.process(
                    self.messages[-1]["role"], self.messages[-1]["content"]
                )
            return json.dumps(self.messages[-1]["content"])
        else:
            return "Invalid role"

    def run_websocket(self):
        self.init_model()
        self.websocket_manager = WebSocketManager(
            self.host, self.port, self._run_websocket
        )
        asyncio.run(self.websocket_manager.init_socket())

    def run_webui(self):
        import streamlit as st

        @st.cache_resource
        def load_model():
            return ModelManager(
                self.model_config,
                self.rag_config,
            )

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
            st_source = st.selectbox(
                "选择书籍", ["红楼梦", "西游记", "三国演义", "水浒传"], key="source"
            )
            self.source = st_source
        with col2:
            st_character = st.selectbox(
                "选择角色",
                [
                    "林黛玉",
                    "贾宝玉",
                    "孙悟空",
                    "猪八戒",
                    "诸葛亮",
                    "曹操",
                    "林冲",
                    "鲁智深",
                ],
                key="char",
            )
            self.character = st_character

        if "messages" not in st.session_state:
            st.session_state.messages = self.messages

        if "audio" not in st.session_state:
            st.session_state.audio = []

        self.model_manager = load_model()

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
                    if self.has_tts:
                        col1, col2 = st.columns([1, 15])
                        with col1:
                            st.markdown(
                                f"<p style='font-size: 36px; color: gray;'> {'🤖'}</p>",
                                unsafe_allow_html=True,
                            )  # 显示表情
                        with col2:
                            # 将音频字节数据编码为base64
                            audio_base64 = base64.b64encode(
                                st.session_state.audio[idx]
                            ).decode("utf-8")
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

                    _ = self.model_manager.infer(
                        st.session_state.messages,
                        self.source,
                        self.character,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        k=numk,
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
                    if self.has_tts:
                        audio_content = self.tts_manager.process(
                            char_dict[self.character],
                            st.session_state.messages[-1]["content"],
                        )
                        # 生成音频

                        st.session_state.audio.append(audio_content)

                    st.rerun()

        # 调用显示函数
        display_messages()
        user_input()

    def run(self):
        if self.isTerminal:
            self.run_terminal()
        elif self.isWebsocket:
            self.run_websocket()
        elif self.isWebUI:
            self.run_webui()
        else:
            raise ValueError("Please specify the running mode: terminal or websocket")

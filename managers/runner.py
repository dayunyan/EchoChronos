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
        source: str,  # 原著
        role: str,  # 角色
        model_name_or_path: str,  # model
        inf_max_length: int,
        adapter_path: str,
        isTerminal: bool,
        isWebsocket: bool,
        tts_server: str,
        host: str,  # websocket
        port: int,
        has_rag: bool,
        **rag_args,
    ):
        self.source = source
        self.role = role
        self.messages = [
            {
                "role": "system",
                "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
            },
        ]
        self.isTerminal = isTerminal
        self.isWebsocket = isWebsocket
        self.host = host
        self.port = port

        self.model_manager = ModelManager(
            model_name_or_path=model_name_or_path,
            inf_max_length=inf_max_length,
            adapter_path=adapter_path,
            has_rag=has_rag,
            **rag_args,
        )
        self.tts_manager = TTSManager(tts_server)

    def run_terminal(self):
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
            self.model_manager.infer(self.messages, self.source, self.role)
            # self.tts_manager.process(
            #     self.messages[-1]["role"], self.messages[-1]["content"]
            # )

    def _run_websocket(self, msg):
        client_msg = json.loads(msg)
        if client_msg["role"] == "user":
            self.messages.append(
                {
                    "role": "user",
                    "content": client_msg["message"],
                }
            )
            self.model_manager.infer(self.messages, self.source, self.role)
            # self.tts_manager.process(
            #     self.messages[-1]["role"], self.messages[-1]["content"]
            # )
            return json.dumps(self.messages[-1]["content"])
        else:
            return "Invalid role"

    def run_websocket(self):
        self.websocket_manager = WebSocketManager(
            self.host, self.port, self._run_websocket
        )
        asyncio.run(self.websocket_manager.init_socket())

    def run(self):
        if self.isTerminal:
            self.run_terminal()
        elif self.isWebsocket:
            self.run_websocket()
        else:
            raise ValueError("Please specify the running mode: terminal or websocket")

import argparse

from managers.constants import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=str,
        default="西游记",
        help="source book or novel or story or anything",
    )
    parser.add_argument(
        "--role",
        type=str,
        default="孙悟空",
        help="role who you want to chat with in source",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="./ChatStyle/.mindnlp/model/Qwen/Qwen2-7B-Instruct",  # Qwen/Qwen2-7B-Instruct
        help="llm model name or path",
    )
    parser.add_argument(
        "--inf_max_length",
        type=int,
        default=128,
        help="max length of inference, recommend 128",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="./ChatStyle/results/20241112LF-qwen2-7B-Instruct",
        help="the path of adapter from finetuning",
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
    parser.add_argument(
        "--tts_server",
        type=str,
        default="http://172.24.244.174:5000/tts",
        help="tts server url",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="websocket host")
    parser.add_argument("--port", type=int, default=6006, help="websocket port")
    parser.add_argument("--has_rag", action="store_true", help="Whether to use RAG")
    parser.add_argument(
        "--base_rag_path",
        type=str,
        default="/root/work_dir/huawei-ict-2024/RAG/rain-rag",
        help="base path of RAG",
    )
    parser.add_argument(
        "--rag_config_path",
        type=str,
        default="config/config.yaml",
        help="config path of RAG",
    )
    parser.add_argument(
        "--rag_embedding_path", type=str, default="", help="embedding path of RAG"
    )
    parser.add_argument(
        "--rag_collection_name",
        type=str,
        default="four_famous",
        help="collection name of RAG",
    )
    return parser.parse_args()


def check_args(args):
    if args.source not in ROLE_DICT.keys():
        raise ValueError("source should be in {}".format(ROLE_DICT.keys()))
    if args.role not in ROLE_DICT[args.source]:
        raise ValueError("role should be in {}".format(ROLE_DICT[args.source]))
    if args.inf_max_length > MAX_INF_LENGTH:
        raise ValueError("inf_max_length should be less than {}".format(MAX_INF_LENGTH))
    if not (args.isTerminal ^ args.isWebsocket):
        raise ValueError("You should choose one of terminal or websocket")

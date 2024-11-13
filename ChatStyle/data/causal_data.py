import json
from tqdm import tqdm
import re
import sys

sys.path.append("../")
from utils.filetools import open_file, save_json


def split_sentence(sentence: str):
    if "“" in sentence and "”" in sentence:
        # 找到所有双引号内的内容，并替换为临时占位符
        inside_quotes = re.findall(r"“(.*?)”", sentence)
        for i, quote in enumerate(inside_quotes):
            sentence = sentence.replace("“" + quote + "”", f"<QUOTE_{i}>")

        # 按照句号分割句子
        sentences = sentence.split("。")

        # 将占位符替换回原来的内容
        for i, quote in enumerate(inside_quotes):
            sentences = [
                (
                    s.replace(f"<QUOTE_{i}>", "“" + quote + "”")
                    if "<QUOTE_" in s
                    else s + "。"
                )
                for s in sentences
            ]
    else:
        sentences = sentence.split("。")
        sentences = [s + "。" for s in sentences if s != ""]
    return sentences


def screen_data(data: list[dict]):
    for i in range(len(data) - 1, -1, -1):
        if (
            "：" not in data[i]["input"]
            or "：" not in data[i]["output"]
            or "“" not in data[i]["output"]
            or "”" not in data[i]["output"]
        ):
            data.pop(i)

    return data


def reconstract_conversation(data: list[dict]):
    conversation = [data[i] for i in range(0, len(data), 2)]
    return conversation


def group_conversation(data: list[dict]):
    new_data = []
    temp = []
    id_temp = data[0]["id"] - 1
    for i in range(len(data)):
        if data[i]["id"] - id_temp == 1:
            temp.append(data[i])
            id_temp = data[i]["id"]
        else:
            new_data.append(reconstract_conversation(temp))
            temp = []
            temp.append(data[i])
            id_temp = data[i]["id"]

    return new_data


if __name__ == "__main__":
    # 读取文件，按行读取
    file_path = "./org/红楼梦.txt"
    text_ls = open_file(file_path)
    text_ls = text_ls[6:]
    print("文本行数：", len(text_ls))
    print("文本内容：", text_ls[:10])
    # 初次整理数据集
    for i in range(len(text_ls) - 1, -1, -1):
        text_ls[i] = text_ls[i].strip("\n,\u3000")
        if text_ls[i][-1] not in ["。", "？", "！", "”"]:
            text_ls[i] += text_ls[i + 1]
            text_ls.pop(i + 1)
    print("文本行数：", len(text_ls))
    print("文本内容：", text_ls[:10])
    # 按行制作数据集
    data = []
    for id, line in tqdm(enumerate(text_ls)):
        line = line.strip("\n,\u3000")
        if not re.search(r"第(.*)回", line):
            data.append({"id": id, "input": "", "output": line})
        else:
            id -= 1

    save_json(data, "./sft/HongLouMeng/HongLouMeng.json")

    # 按“”分割句子：...：“...”
    sentences = []
    temp = ""
    for line in tqdm(text_ls):
        line = line.strip("\n")
        if "“" in line and "”" in line:
            line = line.split("”")
            for i in range(len(line) - 1):
                sentences.append(line[i] + "”")
            if line[-1] != "":
                sentences.append(line[-1])
        elif "“" in line and "”" not in line:
            temp = line
        elif "“" not in line and "”" in line:
            if temp != "":
                sentences.append(temp + line)
                temp = ""
            else:
                sentences.append("“" + line)
        else:
            if temp != "":
                temp += line
            else:
                sentences.append(line)
    print("句子数量：", len(sentences))

    # 按。分割句子（除“”内的句号）
    new_sentences = []
    for sentence in tqdm(sentences):
        new_sentences.extend(split_sentence(sentence))
    print("新句子数量：", len(new_sentences))

    # 制作数据集
    data = []
    for id in tqdm(range(len(new_sentences) - 1)):
        data.append(
            {
                "id": id,
                "instruction": "",
                "input": new_sentences[id],
                "output": new_sentences[id + 1],
            }
        )
    print("制作数据集\n", data[:10])

    # 整理数据集
    data = screen_data(data)
    print("整理数据集1\n", data[:10])
    data = group_conversation(data)
    print("整理数据集2\n", data[:10])

    # 统计
    print("数据集数量：", len(data))
    count = 0
    for i in range(len(data)):
        if len(data[i]) > 5:
            count += 1
    print("对话大于5轮的数据数量：", count)

    save_json(data, "./sft/HongLouMeng/HongLouMeng_Conversation.json")

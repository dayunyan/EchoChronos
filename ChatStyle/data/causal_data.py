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


if __name__ == "__main__":
    # 读取文件，按行读取
    file_path = "./西游记.txt"
    text_ls = open_file(file_path)
    print("文本行数：", len(text_ls))

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
    save_json(data, "./sft/XiYouJi/XiYouJi_Causal.json")

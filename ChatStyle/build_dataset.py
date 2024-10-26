import re
import time
from turtle import st
from typing import List
import json

from numpy import save
from openai import OpenAI


def open_file(path):
    with open(path, "r", encoding="gb18030") as f:
        text = list()
        for line in f:
            if line != "\n":
                text.append(line[:-1])
    f.close()
    return text


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False)
    f.close()


def match_str(pattern, string, method):
    try:
        matchobj = getattr(re, method)(pattern, string)
    except:
        raise AttributeError(f"re does not have attr {method}")
    if matchobj:
        return (matchobj.group(1), matchobj.group(2))
    else:
        return None


def fetch_data(path, max_len, keys: List):
    text_all = open_file(path=path)
    sentence = list()
    que_state = ""
    question = ""
    ans_state = ""
    answer = ""
    constant_key = r"(.*?)道：“(.+?)”"
    for name in keys:
        k = name + constant_key
        for i, text in enumerate(text_all):
            que_state = question = ans_state = answer = ""
            if len(sentence) == max_len - 1:
                return sentence
            matchstr = match_str(k, text, "search")
            if matchstr is not None:
                if len(matchstr[0]) < 10 and len(matchstr[1]) > 0:
                    ans_state = matchstr[0]
                    answer = matchstr[1]
                else:
                    continue

            matchstr = match_str(constant_key, text, "match")
            if matchstr:
                if ans_state != matchstr[0] and answer != matchstr[1]:
                    que_state = matchstr[0]
                    question = matchstr[1]
            if que_state == "" and question == "" and i:
                matchstr = match_str(constant_key, text_all[i - 1], "match")
                if matchstr is not None:
                    if any([_ in matchstr[0] for _ in keys]):
                        # print(matchstr)
                        continue
                    else:
                        que_state = matchstr[0]
                        question = matchstr[1]
            if que_state != "" and question != "" and answer != "":
                sentence.append(
                    {
                        "que_state": que_state,
                        "question": question,
                        "ans_state": ans_state,
                        "answer": answer,
                    }
                )
    print(len(sentence))
    return sentence


def merge_sentence(sentence: List, key):
    string = ""
    for sen in sentence:
        if isinstance(key, List):
            string += sen[key[0]] + "道" + sen[key[1]] + "\n"
        else:
            string += sen[key] + "\n"
    return string


def update_content(sentence, content, start, length):
    content_ls = content.split("\n")
    content_ls = list(filter(lambda x: x != "", content_ls))
    print(f"content_ls: {len(content_ls)}, batchsize: {length}")
    not_update_id = []
    if len(content_ls) < length:
        for i in range(length - len(content_ls)):
            not_update_id.append(start + len(content_ls) + i)
        print(content, content_ls)
    for cont_idx, i in enumerate(range(start, start + length)):
        if cont_idx < len(content_ls):
            sentence[i]["question"] = content_ls[cont_idx]
    return not_update_id


def wyw2bhw(sentence: List):
    client = OpenAI(
        api_key="sk-uQw8AepDDEIs3rJQKyCMZuuMzxYHuy8MfjPTFli1JSN13ImW",
        base_url="https://api.moonshot.cn/v1",
    )
    batchsize = 6
    not_trans_id = []
    for i in range(0, len(sentence), batchsize):
        if (i + batchsize) > len(sentence):
            batchsize = len(sentence) - i
        minibatch = merge_sentence(
            sentence[i : i + batchsize], ["que_state", "question"]
        )
        completion = client.chat.completions.create(
            model="moonshot-v1-8k",
            messages=[
                {
                    "role": "system",
                    "content": "你是一个深通文言文的学者，现在正在做文言文翻译为现代白话文的工作。你要将用户输入的所有语句精确地转换为口语化的白话文，你只需要做这件事，禁止做更多的注解。用户可能会输入多段文字，你需要保证每一段单独转换，段落不要合并。",
                },
                {"role": "user", "content": minibatch},
            ],
            temperature=0.3,
        )
        not_trans = update_content(
            sentence, completion.choices[0].message.content, i, batchsize
        )
        not_trans_id.extend(not_trans)
        # sentence[i]["question"] = completion.choices[0].message.content

        if i % 100 == 0:
            print(sentence[i]["question"])
        time.sleep(30)
    return not_trans_id


if __name__ == "__main__":
    KEYS = ["石猴", "猴王", "悟空", "大圣", "那猴", "行者"]
    sentence = fetch_data("西游记.txt", max_len=100000, keys=KEYS)
    # not_trans_id = wyw2bhw(sentence)
    # print(not_trans_id)
    # save_json("./XiYouJi2.json", sentence)

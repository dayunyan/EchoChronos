import json


def open_file(path):
    with open(path, "r", encoding="gb18030") as f:
        text = list()
        for line in f:
            if line != "\n":
                text.append(line[:-1])
    f.close()
    return text


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False)
    f.close()

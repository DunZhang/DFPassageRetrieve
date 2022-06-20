"""
生成vocab.txt文件
"""
from os.path import join
from collections import defaultdict

# 至少出现3000词的词才放到vocab中
MIN_COUNT = 3000
# 20个unused足以够后面使用了
SPECIAL_TOKENS = ["[CLS]", "[SEP]", "[MASK]", "[UNK]", "[PAD]"] + ["[unused{}]".format(i) for i in range(1, 21)]
if __name__ == "__main__":
    data_dir = "./data"
    word2count = defaultdict(int)

    for name in ["train", "dev", "test"]:
        with open(join(data_dir, "News2022_{}.tsv".format(name)), "r", encoding="utf8") as fr:
            for line in fr:
                for word in line.strip().split("\t")[-1].split(" "):
                    word = word.strip()
                    if len(word) > 0:
                        word2count[word] += 1
    with open("data/News2022_doc.tsv", "r", encoding="utf8") as fr:
        for line in fr:
            for word in line.strip().split("\t")[-1].split(" "):
                word = word.strip()
                if len(word) > 0:
                    word2count[word] += 1
    # 存储
    data = [k for k, v in word2count.items() if v > MIN_COUNT]
    data = SPECIAL_TOKENS + data
    with open(join(data_dir, "vocab.txt"), "w", encoding="utf8") as fw:
        fw.writelines([w + "\n" for w in data])

import random
import torch
from torch.utils.data import Dataset
from Utils.LoggerUtil import LoggerUtil
from transformers import PreTrainedTokenizer
from Config.TrainConfig import TrainConfig
from typing import List, Dict

logger = LoggerUtil.get_logger()


def collect_fn(batch):
    """

    :param batch:List[data_set[i]]
    :return:
    """
    ipt = {}
    query_ids = list(map(lambda x: x[0], batch))
    doc_ids = list(map(lambda x: x[1], batch))

    max_len = max(map(lambda x: len(x), query_ids))
    # vocab里默认4为PAD
    attention_mask = [[1] * len(item) + [0] * (max_len - len(item)) for item in query_ids]
    input_ids = [item + [4] * (max_len - len(item)) for item in query_ids]
    token_type_ids = [[0] * max_len for _ in query_ids]
    ipt["query_ipt"] = {
        "input_ids": torch.LongTensor(input_ids),
        "attention_mask": torch.LongTensor(attention_mask),
        "token_type_ids": torch.LongTensor(token_type_ids),
    }

    # doc的输入
    max_len = max(map(lambda x: len(x), doc_ids))
    # vocab里默认4为PAD
    attention_mask = [[1] * len(item) + [0] * (max_len - len(item)) for item in doc_ids]
    input_ids = [item + [4] * (max_len - len(item)) for item in doc_ids]
    token_type_ids = [[0] * max_len for _ in doc_ids]
    ipt["doc_ipt"] = {
        "input_ids": torch.LongTensor(input_ids),
        "attention_mask": torch.LongTensor(attention_mask),
        "token_type_ids": torch.LongTensor(token_type_ids),
    }
    return ipt


class VectorDataSet(Dataset):
    def __init__(
            self, conf: TrainConfig, data_path: str, tokenizer: PreTrainedTokenizer,
            docid2doc: Dict[str, List[str]], *args, **kwargs):
        """

        :param conf:
        :param data_path:
        :param tokenizer:
        :param docid2doc:
        :param args:
        :param kwargs:
        """
        self.conf = conf
        # 参数初始化
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.docid2doc = docid2doc
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        self.init_data_model()

    def init_data_model(self):
        """ 初始化要用到的模型数据 """
        self.data = []
        with open(self.data_path, "r", encoding="utf8") as fr:
            next(fr)
            for line in fr:
                # (qid, docid, query)
                qid, docid, query = line.strip().split("\t")
                self.data.append([query.strip().split(" ")[:self.conf.max_len], self.docid2doc[docid]])
                # self.data.append([query.strip().split(" ")[:self.conf.max_len], list(self.docid2doc.values())[0]])
        logger.info("总数据{}条".format(len(self.data)))
        random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        """
        item 为数据索引，迭代取第item条数据
        """
        # 获取目标数据
        query, doc = self.data[item]
        query_ids, doc_ids = self.tokenizer.convert_tokens_to_ids(query), self.tokenizer.convert_tokens_to_ids(doc)
        query_ids = [self.tokenizer.cls_token_id] + query_ids + [self.tokenizer.sep_token_id]
        doc_ids = [self.tokenizer.cls_token_id] + doc_ids + [self.tokenizer.sep_token_id]
        return query_ids, doc_ids

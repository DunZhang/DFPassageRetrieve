import numpy as np
import torch
import torch.nn as nn
from Config.TrainConfig import TrainConfig
from roformer import RoFormerModel, RoFormerTokenizer, RoFormerConfig
from typing import Dict
from typing import List, Union
from os.path import join
import torch.nn.functional as F
from Utils.LoggerUtil import LoggerUtil

logger = LoggerUtil.get_logger()


class VectorModel(nn.Module):
    def __init__(self, conf_or_model_dir: Union[str, TrainConfig]):
        """
        如果conf_or_model_dir为配置类则代表从预训练模型开始加载，用于训练
        如果为conf_or_model_dir为目录路径，则从该路径进行加载，该路径下必须有之前存好的模型及相关配置文件，用于继续训练或预测
        """
        super().__init__()
        self.device = None
        # 加载config
        if isinstance(conf_or_model_dir, TrainConfig):
            self.conf = conf_or_model_dir
        else:
            self.conf = TrainConfig()
            self.conf.load(conf_path=join(conf_or_model_dir, "model_conf.yml"))
        # 确定模型目录
        self.pretrained_model_dir = self.conf.pretrained_model_dir
        self.max_len = self.conf.max_len
        # 加载权重
        if isinstance(conf_or_model_dir, TrainConfig):
            # 加载预训练
            CONFIG, TOKENIZER, MODEL = RoFormerConfig, RoFormerTokenizer, RoFormerModel
            self.model = MODEL.from_pretrained(self.pretrained_model_dir)
            self.tokenizer = TOKENIZER.from_pretrained(self.pretrained_model_dir)
            self.backbone_model_config = CONFIG.from_pretrained(self.pretrained_model_dir)
        else:
            # 加载训练好的
            CONFIG, TOKENIZER, MODEL = RoFormerConfig, RoFormerTokenizer, RoFormerModel
            self.tokenizer = TOKENIZER.from_pretrained(conf_or_model_dir)
            self.backbone_model_config = CONFIG.from_pretrained(conf_or_model_dir)
            self.model = MODEL(config=self.backbone_model_config)
            self.load_state_dict(torch.load(join(conf_or_model_dir, "model_weight.bin"), map_location="cpu"))

    def forward(self, ipt: Dict, **kwargs):
        output = {}
        # query
        input_ids = ipt["query_ipt"]["input_ids"].to(self.get_device())
        token_type_ids = ipt["query_ipt"]["token_type_ids"].to(self.get_device())
        attention_mask = ipt["query_ipt"]["attention_mask"].to(self.get_device())
        token_embeddings = self.model(input_ids=input_ids, token_type_ids=token_type_ids,
                                      attention_mask=attention_mask)[0]  # bsz*seq_len*h
        query_vecs = token_embeddings[:, 0, :]  # bsz*h
        output["query_vecs"] = F.normalize(query_vecs, 2.0, dim=1)

        # doc
        input_ids = ipt["doc_ipt"]["input_ids"].to(self.get_device())
        token_type_ids = ipt["doc_ipt"]["token_type_ids"].to(self.get_device())
        attention_mask = ipt["doc_ipt"]["attention_mask"].to(self.get_device())
        token_embeddings = self.model(input_ids=input_ids, token_type_ids=token_type_ids,
                                      attention_mask=attention_mask)[0]  # bsz*seq_len*h
        doc_vecs = token_embeddings[:, 0, :]  # bsz*h
        output["doc_vecs"] = F.normalize(doc_vecs, 2.0, dim=1)
        return output

    def save(self, save_dir):
        self.conf.save(join(save_dir, "model_conf.yml"))
        torch.save(self.state_dict(), join(save_dir, "model_weight.bin"))
        self.backbone_model_config.save_pretrained(save_dir)
        self.tokenizer.save_vocabulary(save_dir)

    def get_sens_vec(self, sens: List[List[str]]):
        # return np.random.random((len(sens), 12)).astype(dtype=np.float32)
        self.model.eval()
        ### get sen vec
        all_sen_vec = []
        start = 0
        with torch.no_grad():
            while start < len(sens):
                # logger.info("get sentences vector: {}/{},\t{}".format(start, len(sens), start / len(sens)))

                input_ids = [
                    self.tokenizer.convert_tokens_to_ids(sen) for sen in sens[start:start + self.conf.batch_size]]
                input_ids = [
                    [self.tokenizer.cls_token_id] + item + [self.tokenizer.sep_token_id] for item in input_ids]
                max_len = max(map(lambda x: len(x), input_ids))
                # vocab里默认4为PAD
                attention_mask = [[1] * len(item) + [0] * (max_len - len(item)) for item in input_ids]
                input_ids = [item + [4] * (max_len - len(item)) for item in input_ids]
                token_type_ids = [[0] * max_len for _ in input_ids]
                input_ids = torch.LongTensor(input_ids).to(self.model.device)
                attention_mask = torch.LongTensor(attention_mask).to(self.model.device)
                token_type_ids = torch.LongTensor(token_type_ids).to(self.model.device)

                token_embeddings = self.model(
                    input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
                pooler_output = token_embeddings[:, 0, :]  # bsz*h
                all_sen_vec.append(pooler_output.to("cpu").numpy())
                start += self.conf.batch_size
        self.model.train()
        return np.vstack(all_sen_vec)

    def get_device(self):
        if self.device is None:
            for v in self.state_dict().values():
                self.device = v.device
                break
        return self.device

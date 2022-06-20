import torch
import torch.nn.functional as F
from Config.TrainConfig import TrainConfig
from typing import Dict


class VectorLossModel(torch.nn.Module):
    def __init__(self, conf: TrainConfig):
        super().__init__()
        self.conf = conf
        self.label = torch.LongTensor(list(range(self.conf.batch_size)))

    def forward(self, model_output: Dict):
        """ batch内负例 """
        Q_vecs = model_output["query_vecs"]  # bsz * h
        T_vecs = model_output["doc_vecs"]  # X * h
        logits = torch.mm(Q_vecs, T_vecs.t())
        loss = F.cross_entropy(logits * self.conf.score_ratio, self.label.to(Q_vecs.device))
        return loss

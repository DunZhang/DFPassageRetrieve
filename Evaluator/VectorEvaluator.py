from typing import Dict, Union, List
from Model.VectorModel import VectorModel
from Config.TrainConfig import TrainConfig
from Utils.LoggerUtil import LoggerUtil
from os.path import join
import faiss

logger = LoggerUtil.get_logger()


class VectorEvaluator():
    def __init__(self, conf: TrainConfig, docid2doc: Dict[str, List[str]]):
        self.conf = conf
        self.data = []
        with open(join(self.conf.data_dir, "News2022_dev.tsv"), "r", encoding="utf8") as fr:
            next(fr)
            for line in fr:
                # (qid, docid, query)
                qid, docid, query = line.strip().split("\t")
                self.data.append([query.strip().split(" ")[:self.conf.max_len], docid2doc[docid]])
                # self.data.append([query.strip().split(" ")[:self.conf.max_len],  list(docid2doc.values())[0]])

    def evaluate(self, model: VectorModel) -> Dict:
        logger.info("evaluate model...")
        query_vecs = model.get_sens_vec(sens=[item[0] for item in self.data])
        doc_vecs = model.get_sens_vec(sens=[item[1] for item in self.data])
        # 构建faiss索引
        faiss_index = faiss.IndexFlatIP(doc_vecs.shape[1])
        faiss_index.add(doc_vecs)
        # 搜索结果
        mrr100 = 0.0000000001
        res_distance, res_index = faiss_index.search(query_vecs, 100)
        for i in range(res_index.shape[0]):
            # 在当前情况下i也是标签
            topk_id = [int(j) for j in res_index[i]]
            if i in topk_id:
                mrr100 += 1 / (1 + topk_id.index(i))
        return {"mrr100": mrr100 / len(self.data)}

    def try_evaluate(self, model: VectorModel, global_step: int) -> Union[Dict, None]:
        if self.conf.eval_step > 1 and global_step % self.conf.eval_step == 0:
            return self.evaluate(model)
        else:
            return None

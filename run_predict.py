import faiss
import torch
from os.path import join
from collections import defaultdict
from Model.VectorModel import VectorModel

if __name__ == "__main__":
    save_path = "predict_result_0620_v2.tsv"
    model_dir = "./output/B256_L256/best-mrr100"
    data_dir = "./data"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VectorModel(conf_or_model_dir=model_dir).to(device)
    # model.conf.batch_size = 512
    conf = model.conf
    # 读取测试集
    qid_list, query_list = [], []
    with open(join(data_dir, "News2022_test.tsv"), "r", encoding="utf8") as fr:
        next(fr)
        for line in fr:
            qid, query = line.strip().split("\t")
            qid_list.append(qid)
            query_list.append(query.strip().split(" "))
    # 读取全量新闻数据
    docid_list, doc_list = [], []
    with open(join(data_dir, "News2022_doc.tsv"), "r", encoding="utf8") as fr:
        next(fr)
        for line in fr:
            docid, doc = line.strip().split("\t")
            doc = doc.strip().split(" ")[:conf.max_len]  # 提前分词并截取最大长度
            docid_list.append(docid)
            doc_list.append(doc)
            # if len(docid_list) > 120000: break

    # 获取所有的问题向量
    print("获取所有的问题向量")
    query_vecs = model.get_sens_vec(sens=query_list)
    # print(query_vecs.shape)
    # 分批获取top-100，避免内存溢出
    start = 0
    res = defaultdict(list)
    while start < len(doc_list):
        print("进度：{}".format(start / len(doc_list)))
        sub_docs = doc_list[start:start + 100000]
        sub_docids = docid_list[start:start + 100000]
        print("获取子向量...")
        doc_vecs = model.get_sens_vec(sens=sub_docs)
        # print(doc_vecs.shape)
        print("faiss 索引...")
        # 构建faiss索引
        faiss_index = faiss.IndexFlatIP(doc_vecs.shape[1])
        faiss_index.add(doc_vecs)
        # 搜索结果
        res_distance, res_index = faiss_index.search(query_vecs, 100)
        for i in range(res_index.shape[0]):
            # 在当前情况下i也是标签
            topk_docid = [sub_docids[int(j)] for j in res_index[i]]
            topk_score = [float(j) for j in res_distance[i]]
            res[qid_list[i]].extend(zip(topk_docid, topk_score))
        start += 100000
    for k, v in res.items():
        v.sort(key=lambda x: x[1], reverse=True)
        res[k] = list(map(lambda x: x[0], v))[:100]
    print("构建预测数据...")
    write_data = [
        "{}\t{}\t{}\n".format(qid, docid, rank) for qid, topk in res.items() for rank, docid in enumerate(topk, 1)]
    print("存储到本地")
    with open(save_path, "w", encoding="utf8") as fw:
        fw.writelines(write_data)

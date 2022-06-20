# DFPassageRetrieve
[新闻文本数据的语义检索与智能问答](https://www.datafountain.cn/competitions/567)的baseline，
2022-06-20排在第3名。

详细的思路请参考博客：https://zhuanlan.zhihu.com/p/531463300
# 运行方法
Step1：获取词表，运行get_vocab.py

Step2：预训练，TF环境，非常耗时，运行pretrain_roformer.py

Step3：预训练模型转为Torch，运行convert_tf_roformer_to_pt.py

Step4：微调，Torch环境，运行run_finetune.py

Step5：预测结果并生成提交文件，Torch+Faiss,运行run_predict.py
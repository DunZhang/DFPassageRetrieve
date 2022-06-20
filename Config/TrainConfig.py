import json
import yaml
from Utils.LoggerUtil import LoggerUtil

logger = LoggerUtil.get_logger()


class TrainConfig():
    def __init__(self):
        # 训练相关
        self.num_epoch = 10
        self.batch_size = 32  #
        self.lr = 5e-5  #
        self.warmup_proportion = 0.1  #
        self.log_step = 20  #
        self.score_ratio = 20.0  # 余弦得分过小，需要放大
        # 模型相关
        self.pretrained_model_dir = ""
        self.device = "0"
        # 评估相关
        self.eval_step = 1000
        self.print_input_step = 200  # 多少步输出一次输入信息
        self.eval_metrics = ["mrr100"]
        # 数据相关
        self.data_dir = "./model_data/sim_data"
        self.max_len = 64  # 最大长度
        # 模型保存相关
        self.output_dir = "./output/test1"
        self.seed = ""
        self.save_times_per_epoch = -1

    def save(self, save_path: str):
        """
        存储配置文件
        :param save_path:
        :return:
        """
        with open(save_path, "w", encoding="utf8") as fw:
            if save_path.endswith(".json"):
                json.dump(self.__dict__, fw, ensure_ascii=False, indent=1)
            else:
                yaml.safe_dump(self.__dict__, fw, encoding="utf8", indent=1, allow_unicode=True)

    def load(self, conf_path: str):
        """
        加载配置文件
        :param conf_path:
        :return:
        """
        with open(conf_path, "r", encoding="utf8") as fr:
            if conf_path.endswith(".json"):
                kwargs = json.load(fr)
            else:
                kwargs = yaml.safe_load(fr)

        for key, value in kwargs.items():
            try:
                if key not in self.__dict__:
                    logger.warning("key:{} 不在类定义中,".format(key))
                setattr(self, key, value)
            except AttributeError as err:
                logger.error("Can't set {} with value {} for {}".format(key, value, self))
                raise err


if __name__ == "__main__":
    TrainConfig().save("../train_configs/conf.yml")

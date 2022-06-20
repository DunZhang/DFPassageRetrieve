import logging
import os

logging.basicConfig(level=logging.INFO)
from Config.TrainConfig import TrainConfig
from Trainer.VectorTrainer import VectorTrainer
from Utils.LoggerUtil import LoggerUtil

if __name__ == "__main__":
    conf = TrainConfig()
    # 指定配置文件即可
    conf.load("./train_configs/conf_B256_L256.yml")
    LoggerUtil.init_logger("Vector", os.path.join(conf.output_dir, "logs.txt"))
    trainer = VectorTrainer(conf)
    trainer.train()

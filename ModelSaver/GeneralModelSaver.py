import os
from os.path import join
from Config.TrainConfig import TrainConfig
from typing import Dict, Union
from Utils.LoggerUtil import LoggerUtil
from Model.VectorModel import VectorModel

logger = LoggerUtil.get_logger()


class GeneralModelSaver():
    def __init__(self, conf: TrainConfig):
        self.conf = conf
        self.best_eval_result = None
        # 创建相关目录
        if not os.path.exists(conf.output_dir):
            os.makedirs(conf.output_dir)
        for metric in conf.eval_metrics:
            if not os.path.exists(join(conf.output_dir, "best-{}".format(metric))):
                os.makedirs(join(conf.output_dir, "best-{}".format(metric)))

    def try_save_model(self, model: VectorModel, global_step: int, epoch_steps: int,
                       eval_result: Union[Dict, None], *args, **kwargs) -> bool:
        """

        :param model:
        :param global_step:
        :param epoch_steps:
        :param eval_result:
        :param args:
        :param kwargs:
        :return:
        """
        if eval_result is not None:
            if self.best_eval_result is None:
                self.best_eval_result = eval_result
                for metric in self.conf.eval_metrics:
                    model.save(join(self.conf.output_dir, "best-{}".format(metric)))
            else:
                for metric in self.conf.eval_metrics:
                    if eval_result[metric] > self.best_eval_result[metric]:
                        logger.info("获取到了更高的:{}指标,存储模型".format(metric))
                        model.save(join(self.conf.output_dir, "best-{}".format(metric)))
                        self.best_eval_result[metric] = eval_result[metric]
        if self.conf.save_times_per_epoch > 0:
            save_step = epoch_steps // self.conf.save_times_per_epoch - 2
            if global_step % save_step == 0:
                save_dir = join(self.conf.output_dir, "step-{}".format(global_step))
                logger.info("save model to:{}".format(save_dir))
                os.makedirs(save_dir)
                model.save(save_dir=save_dir)
                return True
        return False

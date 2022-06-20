import torch
from Config.TrainConfig import TrainConfig
from typing import Dict
from Utils.LoggerUtil import LoggerUtil

logger = LoggerUtil.get_logger()


class VectorInfoLogger():
    def __init__(self, conf: TrainConfig):
        self.conf = conf
        self.best_eval_result = None

    def try_print_log(self, loss: torch.Tensor, eval_result: Dict, step: int, global_step: int, epoch_steps: int,
                      epoch: int, num_epochs: int, *args, **kwargs):
        # 尝试输出loss
        if step % self.conf.log_step == 0:
            print_string = "epoch-{},\tstep:{}/{},\tloss:{}".format(epoch, step, epoch_steps, loss.data)
        else:
            print_string = ""
        if len(print_string) > 5:
            logger.info(print_string)
        if eval_result is not None:
            # 尝试输出本次评测结果以及最优结果
            logger.info("=" * 15 + "本次评测的最终信息" + "=" * 15)
            print_string = ""
            print_string += "本次测评结果是："
            for k, v in eval_result.items():
                print_string += "{}:{},\t".format(k, round(v, 5))
            logger.info(print_string)
            # 判断当前指标是否最好
            if self.best_eval_result is not None:
                for metric in self.conf.eval_metrics:
                    if eval_result[metric] > self.best_eval_result[metric]:
                        logger.info("获取到了更高的指标:{}".format(metric))
                        self.best_eval_result[metric] = eval_result[metric]
                eval_str = "当前最优指标是："
                for k, v in self.best_eval_result.items():
                    eval_str += "{}:{},\t".format(k, round(v, 5))
                logger.info(eval_str)
            else:
                self.best_eval_result = eval_result

        # 输出相关输入信息 主要用于调试
        if self.conf.print_input_step > 0 and global_step % self.conf.print_input_step == 0:
            logger.info("=" * 30 + "随机输出一组输入信息" + "=" * 30)
            ipt = kwargs["ipt"]
            tokenizer = kwargs["tokenizer"]

            ##########################################################################################
            ipt_ids = ipt["query_ipt"]["input_ids"][0].cpu().numpy().tolist()
            token_type_ids = ipt["query_ipt"]["token_type_ids"][0].cpu().numpy().tolist()
            attn_mask = ipt["query_ipt"]["attention_mask"][0].cpu().numpy().tolist()
            ipt_tokens = tokenizer.convert_ids_to_tokens(ipt_ids)
            logger.info("ipt_ids:{}".format(",".join([str(i) for i in ipt_ids])))
            logger.info("ipt_tokens:{}".format(",".join(ipt_tokens)))
            logger.info("token_type_ids:{}".format(",".join([str(i) for i in token_type_ids])))
            logger.info("attn_mask:{}".format(",".join([str(i) for i in attn_mask])))
            ##########################################################################################
            ipt_ids = ipt["doc_ipt"]["input_ids"][0].cpu().numpy().tolist()
            token_type_ids = ipt["doc_ipt"]["token_type_ids"][0].cpu().numpy().tolist()
            attn_mask = ipt["doc_ipt"]["attention_mask"][0].cpu().numpy().tolist()
            ipt_tokens = tokenizer.convert_ids_to_tokens(ipt_ids)
            logger.info("ipt_ids:{}".format(",".join([str(i) for i in ipt_ids])))
            logger.info("ipt_tokens:{}".format(",".join(ipt_tokens)))
            logger.info("token_type_ids:{}".format(",".join([str(i) for i in token_type_ids])))
            logger.info("attn_mask:{}".format(",".join([str(i) for i in attn_mask])))

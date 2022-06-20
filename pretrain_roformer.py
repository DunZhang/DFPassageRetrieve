#! -*- coding: utf-8 -*-
# RoFormerV2 预训练，MLM任务
# nohup ../miniconda3/bin/python pretrain_roformer_v2.py > log_pretrain.txt  2>&1 &

import sys
import os

os.environ['TF_KERAS'] = '1'  # 必须使用tf.keras
import numpy as np
import tensorflow as tf
from shutil import copy
from os.path import join
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.optimizers import extend_with_weight_decay
from bert4keras.optimizers import extend_with_layer_adaptation
from bert4keras.optimizers import extend_with_piecewise_linear_lr
from bert4keras.optimizers import extend_with_gradient_accumulation
from bert4keras.snippets import DataGenerator, parallel_apply_generator

# 基本参数
maxlen = 512
batch_size = 256
epochs = 100
data_dir = "./data"
save_dir = "./output/pretrain_roformer_L-8_H-512_A-8"
steps_per_epoch = 2848580 // batch_size // 4
# 模型配置
config_path = './backbone_model_configs/bert_config.json'
dict_path = './data/vocab.txt'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)


def corpus():
    """语料生成器
    """
    while True:
        for name in ["train", "dev", "test"]:
            with open(join(data_dir, "News2022_{}.tsv".format(name)), "r", encoding="utf8") as fr:
                for line in fr:
                    yield line.strip().split("\t")[-1].strip().split(" ")[:int(maxlen * 1.1)]
        with open("data/News2022_doc.tsv", "r", encoding="utf8") as fr:
            for line in fr:
                yield line.strip().split("\t")[-1].strip().split(" ")[:int(maxlen * 1.1)]


# 加载分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


def mlm_encode(words):
    """WWM语料构建
    """
    rands = np.random.random(len(words))
    source, target = [tokenizer._token_start_id], [0]
    for r, w in zip(rands, words):
        ids = tokenizer.encode(w)[0][1:-1]
        if r < 0.15 * 0.8:
            source.extend([tokenizer._token_mask_id] * len(ids))
            target.extend(ids)
        elif r < 0.15 * 0.9:
            source.extend(ids)
            target.extend(ids)
        elif r < 0.15:
            source.extend(
                np.random.choice(tokenizer._vocab_size - 1, size=len(ids)) + 1
            )
            target.extend(ids)
        else:
            source.extend(ids)
            target.extend([0] * len(ids))
    source = source[:maxlen - 1] + [tokenizer._token_end_id]
    target = target[:maxlen - 1] + [0]
    return source, target


class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, random=False):
        def encode(is_end_text):
            is_end, words = is_end_text
            source, target = mlm_encode(words)
            segment_ids = [0] * len(source)
            return source, segment_ids, target

        for i, d in parallel_apply_generator(
                func=encode,
                iterable=self.sample(random),
                workers=4,
                dummy=sys.platform.startswith("win"),
                max_queue_size=1024
        ):
            yield d


class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉输入部分
    """

    def compute_loss(self, inputs, mask=None):
        y_true, y_pred = inputs
        y_mask = K.cast(K.not_equal(y_true, 0), K.floatx())
        # loss 计算
        loss = K.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=True
        )
        loss = K.sum(loss * y_mask) / (K.sum(y_mask) + K.epsilon())
        # acc 计算
        acc = keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
        acc = K.sum(acc * y_mask) / (K.sum(y_mask) + K.epsilon())
        self.add_metric(acc, name='acc', aggregation='mean')
        # loss 返回
        return loss


strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    base = build_transformer_model(
        config_path,
        checkpoint_path=None,
        model='roformer',
        with_mlm='linear',
        return_keras_model=False
    )
    model = base.model

    # 训练用模型
    y_in = keras.layers.Input(shape=(None,), name='Input-Label')
    outputs = CrossEntropy(1)([y_in, model.output])

    train_model = keras.models.Model(model.inputs + [y_in], outputs)

    AdamW = extend_with_weight_decay(Adam, name='AdamW')
    LAMB = extend_with_layer_adaptation(AdamW, name='LAMB')
    LAMBLR = extend_with_piecewise_linear_lr(LAMB, name='LAMBLR')
    LAMBLRG = extend_with_gradient_accumulation(LAMBLR, name='LAMBLRG')
    optimizer = LAMBLRG(
        learning_rate=1.76e-3,
        bias_correction=False,
        weight_decay_rate=0.01,
        grad_accum_steps=8,
        lr_schedule={
            32000: 1,
            320000: 0.5,
            1280000: 0.1,
            2560000: 0.01
        }
    )
    train_model.compile(optimizer=optimizer)


class Evaluator(keras.callbacks.Callback):
    """训练回调
    """

    def on_epoch_end(self, epoch, logs=None):
        epoch += 1
        base.save_weights_as_checkpoint(join(save_dir, "epoch-{}/bert_model.ckpt".format(epoch)))
        copy(config_path, join(save_dir, "epoch-{}/bert_config.json".format(epoch)))
        copy(dict_path, join(save_dir, "epoch-{}/vocab.txt".format(epoch)))


if __name__ == '__main__':
    # 启动训练
    evaluator = Evaluator()
    train_generator = data_generator(corpus(), batch_size, 10 ** 5)
    dataset = train_generator.to_dataset(
        types=('float32', 'float32', 'float32'),
        shapes=([None], [None], [None]),
        names=('Input-Token', 'Input-Segment', 'Input-Label'),
        padded_batch=True
    )

    train_model.fit(
        dataset, steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=[evaluator]
    )

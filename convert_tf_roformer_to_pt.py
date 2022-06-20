import os
from os.path import join
from shutil import copy
from roformer.convert_roformer_original_tf_checkpoint_to_pytorch import convert_tf_checkpoint_to_pytorch

if __name__ == "__main__":
    tf_model_dir = "./output/pretrain_roformer_L-8_H-512_A-8/epoch-15"
    save_model_dir = "./output/pretrain_roformer_L-8_H-512_A-8/epoch-15/pt"

    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
    convert_tf_checkpoint_to_pytorch(
        tf_checkpoint_path=join(tf_model_dir, "bert_model.ckpt"),
        bert_config_file=join(tf_model_dir, "bert_config.json"),
        pytorch_dump_path=join(save_model_dir, "pytorch_model.bin"))
    copy(src=join(tf_model_dir, "vocab.txt"), dst=join(save_model_dir, "vocab.txt"))
    # TODO 记得自己拷贝一个config到save_model_dir

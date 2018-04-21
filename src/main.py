""" Driver for train """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from .config import get_config, prepare_dirs, save_config
from .data_loader import DataLoader
from .trainer import HMRTrainer


def main(config):
    prepare_dirs(config)

    # Load data on CPU
    with tf.device("/cpu:0"):
        data_loader = DataLoader(config)
        image_loader = data_loader.load()
        smpl_loader = data_loader.get_smpl_loader()

    trainer = HMRTrainer(config, image_loader, smpl_loader)
    save_config(config)
    trainer.train()


if __name__ == '__main__':
    config = get_config()
    main(config)

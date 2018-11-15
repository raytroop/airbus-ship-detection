import os
import sys
import warnings
import logging
import json
import argparse
import numpy as np
import pandas as pd
from skimage.io import imread
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa
from mrcnn import utils
from mrcnn.config import Config
from mrcnn import model as modellib
from utils.rle import rle_decode
from utils.snapshot import SnapshotCallbackBuilder
from utils.logger_wrapper import get_rootlogger
from train import DetectorConfig, DetectorDataset


exclude_list = ['6384c3e78.jpg','13703f040.jpg', '14715c06d.jpg',  '33e0ff2d5.jpg',
                '4d4e09f2a.jpg', '877691df8.jpg', '8b909bb20.jpg', 'a8d99130e.jpg',
                'ad55c3143.jpg', 'c8260c541.jpg', 'd6c7f17c7.jpg', 'dc3e7c901.jpg',
                'e44dffe88.jpg', 'ef87bad36.jpg', 'f083256d8.jpg'] #corrupted images

warnings.filterwarnings('ignore')
def parse_args(args=None):
    # construct the argument parse and parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", default=1, type=int,
                    help="Fold to train")
    parser.add_argument("--backbone", default='resnet50', type=str,
                    help="which fold to train")
    parser.add_argument("--checkpoints", default='logs', type=str,
                    help="Directory to save training logs and trained weights")
    parser.add_argument("--bz", default=8, type=int,
                    help="batch size")
    parser.add_argument('--train', dest='debug', action='store_false',
                            help='Train network | True by default')
    parser.add_argument('--debug', dest='debug', action='store_true',
                            help='Debug network | False by default')
    parser.set_defaults(debug=False)
    parser.add_argument('--nb_epochs', default=20, type=int,
                    help='total number of epochs that the model will be trained for')
    parser.add_argument('--nb_snapshots', default=2, type=int,
                    help='number of times the weights of the model will be saved')
    parser.add_argument('--init_lr', default=0.001, type=float,
                    help='maximum learning rate in cosine anneal schedule ')

    return parser.parse_args(args=args)


if __name__ == '__main__':
    with open('config.json', 'r') as f:
        cfg = json.load(f)
    args = parse_args()

    checkpoints_dir = os.path.join(args.checkpoints, args.backbone,
                           'fold{}'.format(args.fold))

    if not os.path.isdir(checkpoints_dir):
        import errno
        raise FileNotFoundError(
            errno.ENOENT,
            "Could not find model directory under {}".format(checkpoints_dir))

    checkpoints = [f for f in os.listdir(checkpoints_dir) if os.path.splitext(f)[1] == '.h5']
    checkpoint_last = sorted(checkpoints, key = lambda f: int(os.path.splitext(f)[0].split('_')[-1]))[-1]
    checkpoint_last_path = os.path.join(checkpoints_dir, checkpoint_last)

    log_dir = os.path.join(args.checkpoints, args.backbone,
                           'snapshot_fold{}'.format(args.fold))
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    logger = get_rootlogger(os.path.join(log_dir, 'fold{}.log'.format(args.fold)))

    logger.info('Fold {:d} {} snapshot begin'.format(args.fold, args.backbone))
    logger.info('nb_epochs {:2d}, nb_snapshots {:2d}, init_lr {:.5f}'.format(args.nb_epochs, args.nb_snapshots, args.init_lr))
    TRAIN_PATH = cfg['TRAIN_PATH']
    ORIG_SZ = cfg['ORIG_SZ']

    df_trn = pd.read_csv(cfg['TRN_FOLD'].format(args.fold))
    df_val = pd.read_csv(cfg['VAL_FOLD'].format(args.fold))

    trn_fns = df_trn[df_trn.EncodedPixels.notnull()].ImageId.unique().tolist()
    trn_fns = [f for f in trn_fns if trn_fns not in exclude_list]

    val_fns = df_val[df_val.EncodedPixels.notnull()].ImageId.unique().tolist()
    val_fns = [f for f in val_fns if val_fns not in exclude_list]

    logger.info('Prepare train dataset')
    dataset_train = DetectorDataset(
        trn_fns, df_trn, TRAIN_PATH, ORIG_SZ, ORIG_SZ)
    dataset_train.prepare()

    logger.info('Prepare validation dataset')
    dataset_val = DetectorDataset(
        val_fns, df_val, TRAIN_PATH, ORIG_SZ, ORIG_SZ)
    dataset_val.prepare()

    # Image augmentation (light but constant)
    augmentation = iaa.Sequential([
        iaa.OneOf([  # rotate
            iaa.Affine(rotate=0),
            iaa.Affine(rotate=90),
            iaa.Affine(rotate=180),
            iaa.Affine(rotate=270),
        ]),
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([  # brightness or contrast
            iaa.Multiply((0.9, 1.1)),
            iaa.ContrastNormalization((0.9, 1.1)),
        ]),
        iaa.OneOf([  # blur or sharpen
            iaa.GaussianBlur(sigma=(0.0, 0.1)),
            iaa.Sharpen(alpha=(0.0, 0.1)),
        ]),
    ])

    config = DetectorConfig()
    config.BATCH_SIZE = args.bz
    config.IMAGES_PER_GPU = args.bz
    config.BACKBONE = args.backbone

    if not args.debug:
        # 34000 images trn == 500 * 8 * 8.5
        config.STEPS_PER_EPOCH = 150 #len(trn_fns) // args.bz
        # 8500	images val == 250 * 8 * 4.25
        config.VALIDATION_STEPS = 50 #len(val_fns) // args.bz


    model = modellib.MaskRCNN(
        mode='training', config=config, model_dir=log_dir)

    INIT_LR = args.init_lr
    logger.info('Load weight from {}'.format(checkpoint_last_path))
    model.load_weights(checkpoint_last_path, by_name=True)
    callbackBuilder = SnapshotCallbackBuilder(nb_epochs=args.nb_epochs, nb_snapshots=args.nb_snapshots, init_lr=INIT_LR)
    custom_callbacks = callbackBuilder.get_callbacks(log_dir)

    logger.info('initial learning rate {}'.format(args.init_lr))
    logger.info('train {:2d} epochs, {:2d} snapshots'.format(args.nb_epochs, args.nb_snapshots))
    model.train(dataset_train, dataset_val,
                learning_rate=INIT_LR,
                epochs=args.nb_epochs,
                layers='all',
                augmentation=augmentation,
                custom_callbacks=custom_callbacks)
    history = model.keras_model.history.history

    logger.info('save history')
    epochs = range(1, len(history['loss'])+1)
    history_df = pd.DataFrame(history, index=epochs)
    history_df.to_csv(os.path.join(args.checkpoints, args.backbone, 'history_snapshot_fold{}.csv'.format(args.fold)))

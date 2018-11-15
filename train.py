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
from utils.logger_wrapper import get_rootlogger


exclude_list = ['6384c3e78.jpg','13703f040.jpg', '14715c06d.jpg',  '33e0ff2d5.jpg',
                '4d4e09f2a.jpg', '877691df8.jpg', '8b909bb20.jpg', 'a8d99130e.jpg',
                'ad55c3143.jpg', 'c8260c541.jpg', 'd6c7f17c7.jpg', 'dc3e7c901.jpg',
                'e44dffe88.jpg', 'ef87bad36.jpg', 'f083256d8.jpg'] #corrupted images

warnings.filterwarnings('ignore')

def parse_args(args=None):
    """ Parse the arguments.
    """
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

    # args - List of strings to parse. The default is taken from sys.argv.
    return parser.parse_args(args=args)

class DetectorConfig(Config):
    # Give the configuration a recognizable name
    NAME = 'airbus'

    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    BACKBONE = 'resnet50'

    NUM_CLASSES = 2  # background and ship classes

    IMAGE_MIN_DIM = 384
    IMAGE_MAX_DIM = 384
    RPN_ANCHOR_SCALES = (8, 16, 32, 64)
    TRAIN_ROIS_PER_IMAGE = 64
    MAX_GT_INSTANCES = 14
    DETECTION_MAX_INSTANCES = 10
    DETECTION_MIN_CONFIDENCE = 0.95
    DETECTION_NMS_THRESHOLD = 0.0

    # default debug
    STEPS_PER_EPOCH = 15
    VALIDATION_STEPS = 10

    # balance out losses
    LOSS_WEIGHTS = {
        # "rpn_class_loss": 90.0,
        # "rpn_bbox_loss": 1.0,
        # "mrcnn_class_loss": 20.0,
        # "mrcnn_bbox_loss": 3.0,
        # "mrcnn_mask_loss": 3.0
        "rpn_class_loss": 20.0,
        "rpn_bbox_loss": 0.8,
        "mrcnn_class_loss": 6.0,
        "mrcnn_bbox_loss": 1.0,
        "mrcnn_mask_loss": 1.2
    }


class DetectorDataset(utils.Dataset):
    """Dataset class for training our dataset.
    """

    def __init__(self, image_fps, image_annotations, train_dicom_dir, orig_height, orig_width):
        super().__init__(self)

        # Add classes
        self.add_class('ship', 1, 'Ship')

        # add images
        for i, fp in enumerate(image_fps):
            annotations = image_annotations.query(
                'ImageId=="' + fp + '"')['EncodedPixels'].tolist()
            if len(annotations) == 1 and annotations[0] is np.nan:  # empty image
                self.add_image('ship', image_id=i, path=os.path.join(train_dicom_dir, fp),
                               annotations=[], orig_height=orig_height, orig_width=orig_width)
            else:
                self.add_image('ship', image_id=i, path=os.path.join(train_dicom_dir, fp),
                               annotations=annotations, orig_height=orig_height, orig_width=orig_width)

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']

    def load_image(self, image_id):
        info = self.image_info[image_id]
        fp = info['path']
        image = imread(fp)
        # If grayscale. Convert to RGB for consistency.
        if len(image.shape) != 3 or image.shape[2] != 3:
            image = np.stack((image,) * 3, -1)
        return image

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        annotations = info['annotations']
        count = len(annotations)
        if count == 0:
            mask = np.zeros((info['orig_height'], info['orig_width'], 1), dtype=np.uint8)
            class_ids = np.zeros((1,), dtype=np.int32)
        else:
            mask = np.zeros((info['orig_height'], info['orig_width'], count), dtype=np.uint8)
            class_ids = np.zeros((count,), dtype=np.int32)
            for i, a in enumerate(annotations):
                mask[:, :, i] = rle_decode(a)
                class_ids[i] = 1
        return mask.astype(np.bool), class_ids.astype(np.int32)


if __name__ == '__main__':
    with open('config.json', 'r') as f:
        cfg = json.load(f)

    args = parse_args()

    log_dir = os.path.join(args.checkpoints, args.backbone,
                           'fold{}'.format(args.fold))
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    logger = get_rootlogger(os.path.join(log_dir, 'fold{}.log'.format(args.fold)))

    logger.info('Fold {:d} {} begin'.format(args.fold, args.backbone))

    TRAIN_PATH = cfg['TRAIN_PATH']
    ORIG_SZ = cfg['ORIG_SZ']
    COCO_WEIGHTS_PATH = cfg['COCO_WEIGHTS_PATH']
    LEARNING_RATE = cfg['LEARNING_RATE']
    logger.info('initial learning rate {}'.format(LEARNING_RATE))

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
    config.BACKBONE = args.backbone

    if not args.debug:
        # 34000 images trn == 500 * 8 * 8.5
        config.STEPS_PER_EPOCH = 500 #len(trn_fns) // args.bz
        # 8500	images val == 250 * 8 * 4.25
        config.VALIDATION_STEPS = 250 #len(val_fns) // args.bz


    model = modellib.MaskRCNN(
        mode='training', config=config, model_dir=log_dir)

    # Exclude the last layers because they require a matching
    # number of classes
    model.load_weights(COCO_WEIGHTS_PATH, by_name=True, exclude=[
        "mrcnn_class_logits", "mrcnn_bbox_fc",
        "mrcnn_bbox", "mrcnn_mask"])

    logger.info('train to 2 epochs')
    # train heads with higher lr to speedup the learning
    model.train(dataset_train, dataset_val,
                learning_rate=LEARNING_RATE*2,
                epochs=2,
                layers='heads',
                augmentation=None)  # no need to augment yet
    history = model.keras_model.history.history

    logger.info('train to 14 epochs')
    model.train(dataset_train, dataset_val,
                learning_rate=LEARNING_RATE,
                epochs=4 if args.debug else 14,
                layers='all',
                augmentation=augmentation)
    new_history = model.keras_model.history.history
    for k in new_history:
        history[k] = history[k] + new_history[k]

    logger.info('train to 22 epochs')
    model.train(dataset_train, dataset_val,
                learning_rate=LEARNING_RATE/2,
                epochs=6 if args.debug else 22,
                layers='all',
                augmentation=augmentation)
    new_history = model.keras_model.history.history
    for k in new_history:
        history[k] = history[k] + new_history[k]

    logger.info('save history')
    epochs = range(1, len(history['loss'])+1)
    history_df = pd.DataFrame(history, index=epochs)
    history_df.to_csv(os.path.join(args.checkpoints, args.backbone, 'history_fold{}.csv'.format(args.fold)))

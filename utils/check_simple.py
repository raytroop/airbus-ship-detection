import os
import sys
import json
import argparse
import random
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

if __name__ == '__main__' and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    __package__ = "utils"

import mrcnn.model as modellib
from mrcnn import visualize
from train import DetectorConfig, DetectorDataset

exclude_list = ['6384c3e78.jpg','13703f040.jpg', '14715c06d.jpg',  '33e0ff2d5.jpg',
                '4d4e09f2a.jpg', '877691df8.jpg', '8b909bb20.jpg', 'a8d99130e.jpg',
                'ad55c3143.jpg', 'c8260c541.jpg', 'd6c7f17c7.jpg', 'dc3e7c901.jpg',
                'e44dffe88.jpg', 'ef87bad36.jpg', 'f083256d8.jpg'] #corrupted images

def parse_args(args=None):
    """ Parse the arguments.
    """
    # construct the argument parse and parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", default=1, type=int,
                    help="Fold to train")
    parser.add_argument("--backbone", default='resnet50', type=str,
                    help="feature extractor used in mrcnn")
    parser.add_argument("--model_path", default="logs", type=str,
                    help="path/to/logs")

    return parser.parse_args(args=args)

class InferenceConfig(DetectorConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

# set color for class
def get_colors_for_class_ids(class_ids):
    colors = []
    for class_id in class_ids:
        if class_id == 1:
            colors.append((.941, .204, .204))
    return colors


# display mrcnn output

def display_output(model_path, fold, backbone):
    with open('config.json', 'r') as f:
        cfg = json.load(f)
    TRAIN_PATH = cfg['TRAIN_PATH']
    ORIG_SZ = cfg['ORIG_SZ']

    inference_config = InferenceConfig()
    inference_config.BACKBONE = backbone

    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode='inference',
                            config=inference_config,
                            model_dir=os.path.dirname(model_path))

    # Load trained weights (fill in path to trained weights here)
    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    df_val = pd.read_csv("folds/cv5val{}.csv".format(fold))
    val_fns = df_val[df_val.EncodedPixels.notnull()].ImageId.unique().tolist()
    val_fns = [f for f in val_fns if val_fns not in exclude_list]
    dataset_val = DetectorDataset(
        val_fns, df_val, TRAIN_PATH, ORIG_SZ, ORIG_SZ)
    dataset_val.prepare()

    dataset = dataset_val

    fig, axes = plt.subplots(2, 2, figsize=(10, 40))
    fig.tight_layout()
    for i in range(2):

        image_id = random.choice(dataset.image_ids)

        original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset_val, inference_config,
                                image_id, use_mini_mask=False)

    #     print(original_image.shape)
        visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                                    dataset.class_names,
                                    colors=get_colors_for_class_ids(gt_class_id), ax=axes[i, 0])
        # print(original_image.shape, original_image.dtype, original_image.max(), original_image.min())
        # (384, 384, 3) uint8 254 3

        results = model.detect([original_image]) #, verbose=1)
        r = results[0]
        # if len(r['scores']) > 0:
        #     print(r['rois'].shape,r['rois'].dtype, r['rois'].max(), r['rois'].min())
        #     print(r['masks'].shape, r['masks'].dtype, r['masks'].max(), r['masks'].min())
        #     print(r['class_ids'].shape, r['class_ids'].dtype, r['class_ids'].max(), r['class_ids'].min())
        #     print(r['scores'].shape, r['scores'].dtype, r['scores'].max(), r['scores'].min())
        visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                                    dataset.class_names, r['scores'],
                                    colors=get_colors_for_class_ids(r['class_ids']), ax=axes[i, 1])

    axes[0, 0].set_title('Ground True')
    axes[0, 1].set_title('Preditions')

if __name__ == '__main__':
    args = parse_args()
    checkpoints_dir = os.path.join(args.model_path, args.backbone, 'fold{}'.format(args.fold))
    checkpoints = [f for f in os.listdir(checkpoints_dir) if os.path.splitext(f)[1] == '.h5']
    checkpoint_last = sorted(checkpoints, key = lambda f: int(os.path.splitext(f)[0].split('_')[-1]))[-1]
    checkpoint_last_path = os.path.join(checkpoints_dir, checkpoint_last)

    display_output(checkpoint_last_path, args.fold, args.backbone)
    plt.show()



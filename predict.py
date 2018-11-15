import os
import sys
import json
import argparse
import random
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import mrcnn.model as modellib
from mrcnn import visualize
from train import DetectorConfig, DetectorDataset
from utils.ensemble import ensemble


warnings.filterwarnings('ignore')

exclude_list = ['6384c3e78.jpg','13703f040.jpg', '14715c06d.jpg',  '33e0ff2d5.jpg',
                '4d4e09f2a.jpg', '877691df8.jpg', '8b909bb20.jpg', 'a8d99130e.jpg',
                'ad55c3143.jpg', 'c8260c541.jpg', 'd6c7f17c7.jpg', 'dc3e7c901.jpg',
                'e44dffe88.jpg', 'ef87bad36.jpg', 'f083256d8.jpg'] #corrupted images

def parse_args(args):
    parser = argparse.ArgumentParser(description="ensemble model and generate submition")
    parser.add_argument('-o', '--output_name', type=str, required=True, help='output file name')
    parser.add_argument('-w', '--model_weight',  type=str, required=True, help='saved weights to load')
    parser.add_argument('--backbone', default='resnet50', type=str, help='base feature extractor')
    parser.add_argument('--conf_thresh', default=0.5, type=float, help='confidence threshold for valid ship')
    parser.add_argument('--classification', default='assert/ship_detection.csv',
                        help='ship detection https://www.kaggle.com/iafoss/fine-tuning-resnet34-on-ship-detection-new-data/output')
    parser.add_argument('--cls_thresh', default=0.45, type=float, help='classification threshold ship/no-ship')

    return parser.parse_args(args)

def predict_ensemble(image, model, fliplr=True, flipud=True, rot90=True, conf_thresh=0.7, iou_thresh=0.3, nb_mask_pixel=2):
    '''predict masks and bounding box with TTA

    Arguments:
        image: single image, (H, W, 3), uint8
        model: keras model loaded with weight
        fliplr: TTA with flip left/right image, boolean
        flipup: TTA with flip up/down image, boolean
        rot90: TTA with rote 90 degree, boolean
        conf_thresh: confidence threshold in ensemble, float
        iou_thresh: iou threshold in ensemble, float
        nb_mask_pixel: minimum masks for positive pixel

    Returns:
        boxes: list of (y1, x1, y2, x2, conf) - (int, int, int, int, float)
        masks: [H, W, 1] - boolean
    '''
    H, W = image.shape[:2]

    images = [image]
    if fliplr:
        images.append(np.fliplr(image))
    if flipud:
        images.append(np.flipud(image))
    if rot90:
        images.append(np.rot90(image, k=1, axes=(0, 1)))

    results = model.detect(images)

    # rois: int32
    # masks: bool
    # scores: float
    # class_id: int32

    scores = []
    rois = []
    masks = []

    r_org = results.pop(0)
    score = r_org['scores']
    if score.shape[0] > 0:
        scores.append(score)
        rois.append(r_org['rois'])
        masks.append(r_org['masks'])

    if fliplr:
        r = results.pop(0)
        score = r['scores']
        if score.shape[0] > 0:
            scores.append(score)
            roi_lr = r['rois']
            roi_lr = (roi_lr - np.array([[0.0, W, 0.0, W]]))*np.array([[1.0, -1.0, 1.0, -1.0]])
            roi_lr = roi_lr[:, [0, 3, 2, 1]]
            rois.append(roi_lr)

            masks_lr = r['masks']
            masks_lr = np.fliplr(masks_lr)
            masks.append(masks_lr)

    if flipud:
        r = results.pop(0)
        score = r['scores']
        if score.shape[0] > 0:
            scores.append(score)
            roi_ud = r['rois']
            roi_ud = (roi_ud - np.array([[H, 0.0, H, 0.0]]))*np.array([[-1.0, 1.0, -1.0, 1.0]])
            roi_ud = roi_ud[:, [2, 1, 0, 3]]
            rois.append(roi_ud)

            masks_ud = r['masks']
            masks_ud = np.flipud(masks_ud)
            masks.append(masks_ud)

    if rot90:
        r = results.pop(0)
        score = r['scores']
        if score.shape[0] > 0:
            scores.append(score)
            roi_rot = r['rois']
            x1 = roi_rot[:, 1]
            x2 = roi_rot[:, 3]
            y1 = roi_rot[:, 0]
            y2 = roi_rot[:, 2]
            roi_rot = np.stack([x1, H-y2, x2, H-y1]).T
            rois.append(roi_rot)

            masks_rot = r['masks']
            masks_rot = np.rot90(masks_rot, k=1, axes=(1, 0))
            masks.append(masks_rot)

    nb_det = sum([int(x) for x in [fliplr, flipud, rot90]]) + 1
    if scores:
        masks = np.concatenate(masks, -1)
        masks = np.sum(masks, axis=-1, keepdims=True) >= min(nb_mask_pixel, nb_det)
        rois_scores = []
        for roi, score in zip(rois, scores):
            rois_scores.append([list(boxes.astype(int)) + [float(conf)] for boxes, conf in zip(roi, score)])
        new_rois_scores = ensemble(rois_scores, nb_det, conf_thresh=conf_thresh, iou_thresh=iou_thresh) #if len(scores) > 1 else rois_scores[0]
        return new_rois_scores, masks
    return [], np.empty((H, W, 0))

class InferenceConfig(DetectorConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


if __name__ == '__main__':
    args = parse_args()
    with open('config.json', 'r') as f:
        cfg = json.load(f)
    TRAIN_PATH = cfg['TRAIN_PATH']
    ORIG_SZ = cfg['ORIG_SZ']

    inference_config = InferenceConfig()
    inference_config.BACKBONE = 'resnet50'
    inference_config.BATCH_SIZE = 4 # wi TTA
    inference_config.IMAGES_PER_GPU = 4 # wi TTA

    model_path = 'logs/resnet50/fold{}/mask_rcnn_airbus_0022.h5'.format(args.fold)
    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode='inference',
                            config=inference_config,
                            model_dir=os.path.dirname(model_path))

    # Load trained weights (fill in path to trained weights here)
    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    df = pd.read_csv('folds/cv5val{}.csv'.format(args.fold))
    imgfns = df[df.EncodedPixels.notnull()].ImageId.tolist()
    imgfn = random.choice(imgfns)
    img_path = os.path.join(TRAIN_PATH, imgfn)
    img = imread(img_path)
    new_rois_scores, masks = predict_ensemble(img, model, fliplr=True, flipud=True, rot90=True)
    print('==========================')
    print(' y1  x1  y2  x2 \tconf ')
    for found in new_rois_scores:
        print(found)

    if new_rois_scores:
        fig, ax = plt.subplots(figsize=(16, 16))
        visualize.display_masks(img, new_rois_scores, masks, ax)
        plt.show()
    else:
        print("\n*** No instances to display *** \n")







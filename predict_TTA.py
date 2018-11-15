import sys
import os
import gc
import json
import pickle
import logging
import argparse
import numpy as np
import pandas as pd
from skimage.io import imread
from tqdm import tqdm
import mrcnn.model as modellib
from train import DetectorConfig
from utils.rle import rle_encode, multi_rle_encode

exclude_list = ['6384c3e78.jpg','13703f040.jpg', '14715c06d.jpg',  '33e0ff2d5.jpg',
                '4d4e09f2a.jpg', '877691df8.jpg', '8b909bb20.jpg', 'a8d99130e.jpg',
                'ad55c3143.jpg', 'c8260c541.jpg', 'd6c7f17c7.jpg', 'dc3e7c901.jpg',
                'e44dffe88.jpg', 'ef87bad36.jpg', 'f083256d8.jpg'] #corrupted images
def parse_args(args):
    parser = argparse.ArgumentParser(description="ensemble model and generate submition")
    parser.add_argument('-o', '--output_name', type=str, required=True)
    parser.add_argument('-w', '--model_weights', nargs='+', type=str, help='saved weights to load')
    parser.add_argument('--backbone', default='resnet50', type=str, help='base feature extractor')
    parser.add_argument('--conf_thresh', default=0.5, type=float, help='confidence threshold in ensemble')
    parser.add_argument('--iou_thresh', default=0.3, type=float, help='iou threshold in ensemble')
    parser.add_argument('--nb_mask_pixel', default=2, type=int, help='minimum masks for positive pixel')
    parser.add_argument('--fliplr', action='store_true', help='TTA flip left/right | False default')
    parser.add_argument('--flipud', action='store_true', help='TTA flip up/down | False default')
    parser.add_argument('--rot90', action='store_true', help='TTA rotate 90deg | False default')
    parser.add_argument('--classification', default='assert/ship_detection.csv',
                        help='ship detection https://www.kaggle.com/iafoss/fine-tuning-resnet34-on-ship-detection-new-data/output')
    parser.add_argument('--cls_thresh', default=0.45, type=float, help='classification threshold ship/no-ship')

    return parser.parse_args(args)

class InferenceConfig(DetectorConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1	# wi TTA

def detect(image, model, fliplr, flipud, rot90, nb_mask_pixel, conf_thresh):
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
        scores = np.concatenate(scores).reshape(1, 1, -1)
        masks = np.concatenate(masks, -1) * scores
        masks = np.sum(masks, axis=-1, keepdims=True) >= min(nb_mask_pixel, nb_det) * conf_thresh
        return masks
    else:
        return np.full((H, W, 0), False)

def predict_TTA(args=None):
    '''predict masks and bounding box with TTA

    Arguments:
        image: single image, (H, W, 3), uint8
        model: keras model loaded with weight
        fliplr: TTA with flip left/right image, boolean
        flipud: TTA with flip up/down image, boolean
        rot90: TTA with rote 90 degree, boolean
        conf_thresh: confidence threshold in ensemble, float
        iou_thresh: iou threshold in ensemble, float
        nb_mask_pixel: minimum masks for positive pixel

    Returns:
        boxes: list of (y1, x1, y2, x2, conf) - (int, int, int, int, float)
        masks: [H, W, 1] - boolean
    '''

    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

    args = parse_args(args)
    with open('config.json', 'r') as f:
        cfg = json.load(f)
    TST_PATH = cfg['TEST_PATH']
    BATCH_SIZE = sum([int(x) for x in [args.fliplr, args.flipud, args.rot90]]) + 1
    H, W = cfg['ORIG_SZ'], cfg['ORIG_SZ']
    inference_config = InferenceConfig()
    inference_config.BACKBONE = args.backbone
    inference_config.BATCH_SIZE = BATCH_SIZE # wi TTA
    inference_config.IMAGES_PER_GPU = BATCH_SIZE # wi TTA
    model_paths = args.model_weights

    nb_dets = len(model_paths)
    test_names = [f for f in os.listdir(TST_PATH) if f not in exclude_list]
    with open('outs/test_names.pkl', 'wb') as f:
        pickle.dump(test_names, f, -1)
    ship_detection = pd.read_csv(args.classification, index_col='id')
    test_names_nothing = ship_detection.loc[ship_detection['p_ship'] <= args.cls_thresh].index.tolist()

    results  = []
    for i, model_path in enumerate(model_paths):
        model = modellib.MaskRCNN(mode='inference',
                                  config=inference_config,
                                  model_dir=os.path.dirname(model_path))
        assert model_path != "", "Provide path to trained weights"
        logging.info("Loading weights from " + model_path)
        model.load_weights(model_path, by_name=True)
        logging.info("model {} begin predict".format(i))

        masks_gather = []
        for image_id in tqdm(test_names):
            if image_id not in test_names_nothing:
                image = imread(os.path.join(TST_PATH, image_id))
                masks_gather.append(detect(image, model, args.fliplr, args.flipud, args.rot90, args.nb_mask_pixel, args.conf_thresh))
            else:
                masks_gather.append(np.full((H, W, 0), False))

        results.append(masks_gather)
        del model
        gc.collect()

        pkl_nme = 'model_{}_preds.pkl'.format(9)
        logging.info('{} saving'.format(pkl_nme))
        pkl_file = os.path.join('outs', pkl_nme)
        with open(pkl_file, 'wb') as f:
            pickle.dump(masks_gather, f, -1)
        logging.info("model {} done".format(i))

    submission_file = os.path.join('outs', args.output_name+'.csv')
    with open(submission_file, 'w') as file:
        logging.info("submission saved as " + submission_file)
        file.write("ImageId,EncodedPixels\n")
        i = 0
        for masks in tqdm(zip(*(results)), total=len(test_names)):
            image_id = test_names[i]
            masks_concat = np.concatenate(masks, axis=-1)
            if masks_concat.shape[-1] > 0:
                masks_concat = np.sum(masks_concat, -1) > (nb_dets // 2)
                labels = multi_rle_encode(masks_concat)
                if labels:
                    for label in labels:
                        file.write(image_id + "," + label + "\n")
                else:
                    file.write(image_id + ",\n")  ## no ship
            else:
                    file.write(image_id + ",\n")  ## no ship
            i += 1


if __name__ == '__main__':
    predict_TTA()

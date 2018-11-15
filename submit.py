import sys
import os
import json
import argparse
import numpy as np
import pandas as pd
from skimage.io import imread
from tqdm import tqdm
import mrcnn.model as modellib
from predict import predict_ensemble
from train import DetectorConfig
from utils.rle import multi_rle_encode
from utils.ensemble import ensemble


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


def submit_enseble(args=None):
    args = parse_args(args)
    with open('config.json', 'r') as f:
        cfg = json.load(f)
    TST_PATH = cfg['TEST_PATH']
    BATCH_SIZE = sum([int(x) for x in [args.fliplr, args.flipud, args.rot90]]) + 1

    inference_config = InferenceConfig()
    inference_config.BACKBONE = args.backbone
    inference_config.BATCH_SIZE = BATCH_SIZE # wi TTA
    inference_config.IMAGES_PER_GPU = BATCH_SIZE # wi TTA

    model_weights = args.model_weights
    nb_det = len(model_weights)
    model_ensemble = []
    for model_path in model_weights:
        model = modellib.MaskRCNN(mode='inference',
                                config=inference_config,
                                model_dir=os.path.dirname(model_path))
        assert model_path != "", "Provide path to trained weights"
        print("Loading weights from ", model_path)
        model.load_weights(model_path, by_name=True)
        model_ensemble.append(model)

    test_names = [f for f in os.listdir(TST_PATH) if f not in exclude_list]
    ship_detection = pd.read_csv(args.classification, index_col='id')
    test_names_nothing = ship_detection.loc[ship_detection['p_ship'] <= args.cls_thresh].index.tolist()

    with open(args.output_name+'.csv', 'w') as file:
        file.write("ImageId,EncodedPixels\n")
        for image_id in tqdm(test_names):
            if image_id not in test_names_nothing:
                image = imread(os.path.join(TST_PATH, image_id))
                # If grayscale. Convert to RGB for consistency.
                if len(image.shape) != 3 or image.shape[2] != 3:
                    image = np.stack((image,) * 3, -1)

                rois_scores_gather = []
                masks_gather = []
                n = 0
                for model in model_ensemble:
                    rois_scores, masks = predict_ensemble(image, model, fliplr=args.fliplr, flipud=args.flipud, rot90=args.rot90, conf_thresh=args.conf_thresh,
                                               iou_thresh=args.iou_thresh, nb_mask_pixel=args.nb_mask_pixel)
                    if rois_scores:
                        n += 1
                        rois_scores_gather.append(rois_scores)
                        masks_gather.append(masks)
                if n > 0:
                    rois_scores_gather = ensemble(
                        rois_scores_gather, nb_det, conf_thresh=args.conf_thresh*0.8, iou_thresh=args.iou_thresh) #if n > 1 else rois_scores_gather[0]
                    masks_gather = np.concatenate(masks_gather, axis=-1)
                    masks_gather = np.sum(masks_gather, axis=-1) >= min(args.nb_mask_pixel, nb_det)
                    # print(rois_scores_gather)
                    for y1, x1, y2, x2, _ in rois_scores_gather:
                        masks_gather[y1:y2, x1:x2] = True

                    labels = multi_rle_encode(masks_gather)
                    if labels:
                        for label in labels:
                            file.write(image_id + "," + label + "\n")
                    else:
                        file.write(image_id + ",\n")  ## no ship
                else:
                    file.write(image_id + ",\n")  ## no ship
            else:
                file.write(image_id + ",\n")  ## no ship


if __name__ == '__main__':
    submit_enseble()

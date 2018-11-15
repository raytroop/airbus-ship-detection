import sys
import os
import json
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
    parser.add_argument('-o', '--output_name', type=str, required=True, help='output file name')
    parser.add_argument('-w', '--model_weight',  type=str, required=True, help='saved weights to load')
    parser.add_argument('--backbone', default='resnet50', type=str, help='base feature extractor')
    parser.add_argument('--conf_thresh', default=0.5, type=float, help='confidence threshold for valid ship')
    parser.add_argument('--classification', default='assert/ship_detection.csv',
                        help='ship detection https://www.kaggle.com/iafoss/fine-tuning-resnet34-on-ship-detection-new-data/output')
    parser.add_argument('--cls_thresh', default=0.45, type=float, help='classification threshold ship/no-ship')

    return parser.parse_args(args)

class InferenceConfig(DetectorConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1	# wi TTA

def predict(args=None):
    args = parse_args(args)
    with open('config.json', 'r') as f:
        cfg = json.load(f)
    TST_PATH = cfg['TEST_PATH']
    model_path = args.model_weight

    inference_config = InferenceConfig()
    inference_config.BACKBONE = args.backbone
    model = modellib.MaskRCNN(mode='inference',
                                config=inference_config,
                                model_dir=os.path.dirname(model_path))
    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    test_names = [f for f in os.listdir(TST_PATH) if f not in exclude_list]
    ship_detection = pd.read_csv(args.classification, index_col='id')
    test_names_nothing = ship_detection.loc[ship_detection['p_ship'] <= args.cls_thresh].index.tolist()

    with open(args.output_name+'.csv', 'w') as file:
        file.write("ImageId,EncodedPixels\n")

        for image_id in tqdm(test_names):
            found = False

            if image_id not in test_names_nothing:
                image = imread(os.path.join(TST_PATH, image_id))
                # If grayscale. Convert to RGB for consistency.
                if len(image.shape) != 3 or image.shape[2] != 3:
                    image = np.stack((image,) * 3, -1)

                results = model.detect([image])
                r = results[0]

                assert( len(r['rois']) == len(r['class_ids']) == len(r['scores']) )
                if len(r['rois']) == 0:
                    pass  ## no ship
                else:
                    num_instances = len(r['rois'])

                    for i in range(num_instances):
                        if r['scores'][i] > args.conf_thresh:
                            file.write(image_id + "," + rle_encode(r['masks'][...,i]) + "\n")
                            found = True

            if not found:
                file.write(image_id + ",\n")  ## no ship

if __name__ == '__main__':
    predict()
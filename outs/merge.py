import sys
import os
import gc
import logging
import pickle
from tqdm import trange
import json
import numpy as np
import pandas as pd


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.rle import multi_rle_encode

with open('config.json', 'r') as f:
    cfg = json.load(f)
H, W = cfg['ORIG_SZ'], cfg['ORIG_SZ']
N = 15606
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

container = np.zeros((N, H, W), dtype=np.int8)
for i in range(10):
    logging.info('Load model {} predtions'.format(i))
    with open('outs/model_{}_preds.pkl'.format(i), 'rb') as f:
        result = pickle.load(f)

    logging.info('preprocess model {} predtions'.format(i))
    for i in trange(N):
        if result[i].shape[-1] == 0:
            continue
        else:
            container[i] += result[i][..., 0].astype(np.int8)

np.save('outs/merged', container)

logging.info('Load test names')
with open('outs/test_names.pkl', 'rb') as f:
    test_names = pickle.load(f)

assert len(test_names) == N, 'test names should be same with preditions'

submission_file = 'outs/submission_merge.csv'
logging.info("submission saved as " + submission_file)
with open(submission_file, 'w') as file:
    file.write("ImageId,EncodedPixels\n")
    logging.info('Begin write submission file')
    for i in trange(N):
        image_id = test_names[i]
        mask = container[i]
        mask = mask > 5	# greater than half of predictor
        labels = multi_rle_encode(mask)
        if labels:
            for label in labels:
                file.write(image_id + "," + label + "\n")
        else:
            file.write(image_id + ",\n")  ## no ship

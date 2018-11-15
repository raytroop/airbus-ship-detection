import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def parse_args(args):
    """ Parse the arguments.
    """
    # construct the argument parse and parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", default=1, type=int,
                    help="Fold to train")
    parser.add_argument("--logdir", default='logs/resnet50', type=str,
                    help="which fold to train")
    # args - List of strings to parse. The default is taken from sys.argv.
    return parser.parse_args(args=args)

def plot_history(args=None):
    args = parse_args(args)
    fps = os.path.join(args.logdir, 'history_fold{}.csv'.format(args.fold))
    df = pd.read_csv(fps)
    df = df.clip_upper(5.0)
    history = df.to_dict('list')
    epochs = range(1, len(history['loss'])+1)

    plt.figure(figsize=(21,11))
    plt.subplot(231)
    plt.plot(epochs, history["loss"], label="Train loss")
    plt.plot(epochs, history["val_loss"], label="Valid loss")
    plt.legend()
    plt.subplot(232)
    plt.plot(epochs, history["rpn_class_loss"], label="Train RPN class ce")
    plt.plot(epochs, history["val_rpn_class_loss"], label="Valid RPN class ce")
    plt.legend()
    plt.subplot(233)
    plt.plot(epochs, history["rpn_bbox_loss"], label="Train RPN box loss")
    plt.plot(epochs, history["val_rpn_bbox_loss"], label="Valid RPN box loss")
    plt.legend()
    plt.subplot(234)
    plt.plot(epochs, history["mrcnn_class_loss"], label="Train MRCNN class ce")
    plt.plot(epochs, history["val_mrcnn_class_loss"], label="Valid MRCNN class ce")
    plt.legend()
    plt.subplot(235)
    plt.plot(epochs, history["mrcnn_bbox_loss"], label="Train MRCNN box loss")
    plt.plot(epochs, history["val_mrcnn_bbox_loss"], label="Valid MRCNN box loss")
    plt.legend()
    plt.subplot(236)
    plt.plot(epochs, history["mrcnn_mask_loss"], label="Train Mask loss")
    plt.plot(epochs, history["val_mrcnn_mask_loss"], label="Valid Mask loss")
    plt.legend()

    plt.show()

if __name__ == '__main__':
    plot_history()
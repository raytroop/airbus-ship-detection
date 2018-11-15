#!/usr/bin/env bash

# python predict_TTA.py -o submission_4rd_TTA	-w logs/resnet50/snapshot_fold1/mask_rcnn_airbus_Best.h5 \
#                                             --fliplr --flipud --rot90 --conf_thresh 0.5


# python predict_TTA.py -o submission_cv_TTA	-w  logs/resnet50/snapshot_fold1/mask_rcnn_airbus_Best.h5 \
#                                                 logs/resnet50/snapshot_fold2/mask_rcnn_airbus_Best.h5 \
#                                                 logs/resnet50/snapshot_fold3/mask_rcnn_airbus_Best.h5 \
#                                                 logs/resnet50/snapshot_fold4/mask_rcnn_airbus_Best.h5 \
#                                                 logs/resnet50/snapshot_fold5/mask_rcnn_airbus_Best.h5 \
#                                                 --fliplr --flipud --rot90 --conf_thresh 0.7


# python predict_TTA.py -o submission_cv_clr_TTA	-w  logs/resnet50/snapshot_fold1/mask_rcnn_airbus_1.h5 \
#                                                     logs/resnet50/snapshot_fold2/mask_rcnn_airbus_1.h5 \
#                                                     logs/resnet50/snapshot_fold3/mask_rcnn_airbus_1.h5 \
#                                                     logs/resnet50/snapshot_fold4/mask_rcnn_airbus_1.h5 \
#                                                     logs/resnet50/snapshot_fold5/mask_rcnn_airbus_1.h5 \
#                                                     logs/resnet50/snapshot_fold1/mask_rcnn_airbus_2.h5 \
#                                                     logs/resnet50/snapshot_fold2/mask_rcnn_airbus_2.h5 \
#                                                     logs/resnet50/snapshot_fold3/mask_rcnn_airbus_2.h5 \
#                                                     logs/resnet50/snapshot_fold4/mask_rcnn_airbus_2.h5 \
#                                                     logs/resnet50/snapshot_fold5/mask_rcnn_airbus_2.h5 \
#                                                     --fliplr --flipud --rot90 --conf_thresh 0.7

python predict_TTA.py -o submission_cv_clr_TTA	-w  logs/resnet50/snapshot_fold5/mask_rcnn_airbus_2.h5 \
                                                    --fliplr --flipud --rot90 --conf_thresh 0.7
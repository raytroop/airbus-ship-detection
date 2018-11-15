#!/usr/bin/env bash

# python submit.py -o submission_2nd	--model_weights assert/mask_rcnn_airbus_0022.h5
#                                         # logs/resnet50/fold1/mask_rcnn_airbus_0022.h5 \
#                                         # logs/resnet50/fold2/mask_rcnn_airbus_0022.h5 \
#                                         # logs/resnet50/fold3/mask_rcnn_airbus_0022.h5 \
#                                         # logs/resnet50/fold4/mask_rcnn_airbus_0022.h5 \
#                                         # logs/resnet50/fold5/mask_rcnn_airbus_0022.h5


python submit.py -o submission_3rd	-w logs/resnet50/snapshot_fold1/mask_rcnn_airbus_Best.h5 \
                                        logs/resnet50/snapshot_fold2/mask_rcnn_airbus_Best.h5 \
                                        logs/resnet50/snapshot_fold3/mask_rcnn_airbus_Best.h5 \
                                        logs/resnet50/snapshot_fold4/mask_rcnn_airbus_Best.h5 \
                                        logs/resnet50/snapshot_fold5/mask_rcnn_airbus_Best.h5 \
                                        --fliplr --flipud --rot90

"""
Based on Ensembling methods for object detection.

https://github.com/ahrnbom/ensemble-objdet.git
"""
def calculate_iou(box1, box2):
    y11, x11, y12, x12 = box1
    y21, x21, y22, x22 = box2
    w1 = x12 - x11
    h1 = y12 - y11
    w2 = x22 - x21
    h2 = y22 - y21

    assert w1 * h1 > 0
    assert w2 * h2 > 0

    area1, area2 = w1 * h1, w2 * h2
    xi1, yi1, xi2, yi2 = max([x11, x21]), max([y11, y21]), min([x12, x22]), min([y12, y22])

    if xi2 <= xi1 or yi2 <= yi1:
        return 0

    intersect = (xi2-xi1) * (yi2-yi1)
    union = area1 + area2 - intersect
    return intersect / union

"""
Ensemble - find overlapping boxes of the same class and average their positions
while adding their confidences. Can weigh different detectors with different weights.
No real learning here, although the weights and iou_thresh can be optimized.

Input:
 - dets : List of detections. Each detection is all the output from one detector, and
          should be a list of boxes, where each box should be on the format
          [box_y1, box_x1, box_y2, box_x2, confidence] where box_y1 and box_x1
          are the upper left coordinates, box_y2 and box_x2 are lower right resp.
          The values should be floats.

 - iou_thresh: Threshold in terms of IOU where two boxes are considered the same,
               if they also belong to the same class.

 - weights: A list of weights, describing how much more some detectors should
            be trusted compared to others. The list should be as long as the
            number of detections. If this is set to None, then all detectors
            will be considered equally reliable. The sum of weights does not
            necessarily have to be 1.

Output:
    A list of boxes, on the same format as the input. Confidences are in range 0-1.
"""
def ensemble(valid_dets, ndets, conf_thresh=0.5, iou_thresh=0.3, weights=None):
    if len(valid_dets) == 1 and ndets > 1:
        return list()
    if len(valid_dets) == 1 and ndets == len(valid_dets):
        return valid_dets

    assert(type(iou_thresh) == float)

    if weights is None:
        w = 1/float(ndets)
        weights = [w]*ndets
    else:
        assert(len(weights) == ndets)

        s = sum(weights)
        for i in range(len(weights)):
            weights[i] /= s

    out = list()
    used = set()
    npred = len(valid_dets)
    for idet in range(npred):
        det = valid_dets[idet]
        for box in det:
            if tuple(box) in used:
                continue

            used.add(tuple(box))
            # Search the other detectors for overlapping box of same class
            found = []
            for iodet in range(npred):
                odet = valid_dets[iodet]

                if odet == det:
                    continue

                bestbox = None
                bestiou = iou_thresh
                for obox in odet:
                    if not tuple(obox) in used:
                        # Not already used
                        # Same class
                        iou = calculate_iou(box[:4], obox[:4])
                        if iou > bestiou:
                            bestiou = iou
                            bestbox = obox

                if not bestbox is None:
                    w = weights[iodet]
                    found.append((bestbox, w))
                    used.add(tuple(bestbox))

            # Now we've gone through all other detectors
            if not found:
                continue
                # new_box = list(box)
                # new_box[4] /= ndets
                # # never append
                # if new_box[4] >= conf_thresh:
                #     out.append(new_box)
            else:
                allboxes = [(box, weights[idet])]
                allboxes.extend(found)

                by1 = 0.0
                bx1 = 0.0
                by2 = 0.0
                bx2 = 0.0
                conf = 0.0

                wsum = 0.0
                for bb in allboxes:

                    # (box_x, box_y, box_w, box_h)
                    b = bb[0]

                    by1 += w*b[4]*b[0]
                    bx1 += w*b[4]*b[1]
                    by2 += w*b[4]*b[2]
                    bx2 += w*b[4]*b[3]
                    conf += w*b[4]
                    # weight
                    w = bb[1]
                    wsum += w*b[4]

                by1 /= wsum
                bx1 /= wsum
                by2 /= wsum
                bx2 /= wsum

                new_box = [int(by1), int(bx1), int(by2), int(bx2), conf]
                if new_box[4] >= conf_thresh:
                    out.append(new_box)
    return out

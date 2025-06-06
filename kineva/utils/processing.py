"""
Kineva Inference Engine
Copyright (C) Rebotnix

Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
For license details, see: https://www.gnu.org/licenses/agpl-3.0.html
Project website: https://rebotnix.com
"""

import cv2
import time
import numpy as np
from PIL import Image
import kornia as K
import torch
import torchvision
import torchvision.transforms.functional as F
import torch.nn.functional as FF
from torchvision.transforms.functional import pil_to_tensor


def is_box_inside(box1, box2):
    # box format: [x1, y1, x2, y2]
    # returns True if box1 is fully inside box2
    return (box1[0] >= box2[0] and box1[1] >= box2[1] and
            box1[2] <= box2[2] and box1[3] <= box2[3])

def remove_nested_boxes(boxes, scores, iou_threshold=0.7):
    keep = []
    idxs = scores.argsort(descending=True)
    for i in idxs:
        box_i = boxes[i]
        suppress = False
        for j in keep:
            box_j = boxes[j]
            # IoU calculation
            inter_x1 = max(box_i[0], box_j[0])
            inter_y1 = max(box_i[1], box_j[1])
            inter_x2 = min(box_i[2], box_j[2])
            inter_y2 = min(box_i[3], box_j[3])
            inter_w = max(0, inter_x2 - inter_x1)
            inter_h = max(0, inter_y2 - inter_y1)
            inter_area = inter_w * inter_h

            area_i = (box_i[2] - box_i[0]) * (box_i[3] - box_i[1])
            area_j = (box_j[2] - box_j[0]) * (box_j[3] - box_j[1])
            union_area = area_i + area_j - inter_area

            iou = inter_area / union_area if union_area > 0 else 0

            # Also check if box_i is fully inside box_j
            inside = is_box_inside(box_i, box_j)

            if iou > iou_threshold or inside:
                suppress = True
                break
        if not suppress:
            keep.append(i.item())
    return torch.tensor(keep, device=boxes.device)

def postprocess_kineva(prediction, ratios, num_classes, conf_thre=0.35, nms_thre=0.7, class_agnostic=False, obj_conf_enabled=True):
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    if not obj_conf_enabled:
        prediction[..., 4] = 1

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float(), image_pred[:, 5 + num_classes:]), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        if class_agnostic:
            nms_out_index = torchvision.ops.nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                nms_thre,
            )
        else:
            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                nms_thre,
            )

        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    for i, r in enumerate(ratios):
        if output[i] is not None:
            output[i] = output[i].cpu()
            output[i][..., :4] /= r

    detections = output[0]  # Get the actual tensor from the list

    final_boxes = detections[:, 0:4]     # x1, y1, x2, y2
    final_scores = detections[:, 4]      # confidence
    final_labels = detections[:, 6].long()  # class_id as integers

    return final_boxes, final_scores, final_labels

def agnostic_nms_with_nested_removal(outputs_data, original_image, conf_threshold=0.35, iou_threshold=0.7, max_detections=300):

    raw_boxes = torch.from_numpy(outputs_data[0])
    raw_logits = torch.from_numpy(outputs_data[1])


    probs = FF.softmax(raw_logits, dim=-1)
    scores, labels = probs.max(-1)

    raw_boxes = raw_boxes[0][:]

    img_h, img_w, img_c = original_image.shape  
    # Scale from normalized [0, 1] format to pixel coordinates [x1, y1, x2, y2]
    boxes_xywh = raw_boxes.clone()
    boxes_xywh[:, 0] *= img_w   # cx
    boxes_xywh[:, 1] *= img_h  # cy
    boxes_xywh[:, 2] *= img_w   # w
    boxes_xywh[:, 3] *= img_h  # h

    # Convert from [cx, cy, w, h] → [x1, y1, x2, y2]
    boxes = torch.zeros_like(boxes_xywh)
    boxes[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2  # x1
    boxes[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2  # y1
    boxes[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2  # x2
    boxes[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2  # y2

    scores = scores.view(-1)
    labels = labels.view(-1)
    device = boxes.device

    # Filter by confidence
    keep_conf = scores > conf_threshold
    boxes = boxes[keep_conf]
    scores = scores[keep_conf]
    labels = labels[keep_conf]

    if boxes.numel() == 0:
        return (torch.empty((0, 4), device=device),
                torch.empty((0,), device=device),
                torch.empty((0,), dtype=torch.long, device=device))

    # Sort by descending score
    scores_sorted, idxs = scores.sort(descending=True)
    boxes_sorted = boxes[idxs]
    labels_sorted = labels[idxs]

    # Apply standard class-agnostic NMS
    keep = torchvision.ops.nms(boxes_sorted, scores_sorted, iou_threshold)

    # Keep only max_detections
    keep = keep[:max_detections]

    # Filter nested boxes inside kept boxes
    kept_boxes = boxes_sorted[keep]
    kept_scores = scores_sorted[keep]
    kept_labels = labels_sorted[keep]

    final_keep = remove_nested_boxes(kept_boxes, kept_scores, iou_threshold)
    
    # Map back to original indices after nested removal
    final_keep = keep[final_keep]

    final_boxes = boxes_sorted[final_keep]
    final_scores = scores_sorted[final_keep]
    final_labels = labels_sorted[final_keep]

    return final_boxes, final_scores, final_labels

def preprocess(image_path, img_size=560, mode="anomaly", means=[0.485, 0.456, 0.406], stds=[0.229, 0.224, 0.225]):
    if mode =="anomaly":
        if isinstance(image_path, str):
            tensor = read_image_as_tensor(image_path)
            if tensor.shape[2] != img_size or tensor.shape[3] != img_size:
                tensor = K.augmentation.Resize([img_size, img_size])(tensor)
            return tensor.numpy().astype(np.float32)
        else:
            print("cv2")
            tensor = read_cv2_image_as_tensor(image_path)
            if tensor.shape[2] != img_size or tensor.shape[3] != img_size:
                tensor = K.augmentation.Resize([img_size, img_size])(tensor)
            return tensor.numpy().astype(np.float32)
        
    elif mode == "kineva":
        img = cv2.imread(image_path)
        orig = img.copy()
        imgs = [img]
        pad_ims = []
        rs = []
        swap=(2, 0, 1)
        for img in imgs:
            if len(img.shape) == 3:
                padded_img = np.ones((img_size[0], img_size[1], 3), dtype=np.uint8) * 114
            else:
                padded_img = np.ones(img_size, dtype=np.uint8) * 114

            r = min(img_size[0] / img.shape[0], img_size[1] / img.shape[1])
            resized_img = cv2.resize(
                img,
                (int(img.shape[1] * r), int(img.shape[0] * r)),
                interpolation=cv2.INTER_LINEAR,
            ).astype(np.uint8)
            padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

            padded_img = padded_img.transpose(swap)
            padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
            pad_ims.append(torch.from_numpy(padded_img).unsqueeze(0))
            rs.append(r)

        ret_ims = pad_ims[0] if len(pad_ims) == 1 else torch.cat(pad_ims)

        return ret_ims, rs

    
    elif mode == "rfdetr":
        orig_sizes = []
        processed_images = []
        if isinstance(image_path, str):
            img_tensor = read_image_as_tensor(image_path)
            h, w = img_tensor.shape[1:]
            orig_sizes.append((h, w))
            img_tensor = F.normalize(img_tensor, means, stds)
            img_tensor = F.resize(img_tensor, (img_size, img_size)).numpy().astype(np.float32)
        
            return img_tensor[None, ...]

    elif mode == "ultralytics":
        img = cv2.imread(image_path)
        orig = img.copy()
        h0, w0 = img.shape[:2]
        r = min(img_size[0] / h0, img_size[1] / w0)
        new_w, new_h = int(w0 * r), int(h0 * r)
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((img_size[0], img_size[1], 3), 114, dtype=np.uint8)
        dw = (img_size[1] - new_w) // 2
        dh = (img_size[0] - new_h) // 2
        canvas[dh:dh + new_h, dw:dw + new_w, :] = img_resized

        # BGR to RGB, HWC to CHW
        img = canvas[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0

        # Add batch dimension
        img = np.expand_dims(img, axis=0)  # (1, 3, H, W)
        return img, orig
    
def read_image_as_tensor(path_to_image):
    pil_image = Image.open(path_to_image).convert("RGB")
    image_as_tensor = pil_to_tensor(pil_image).float() / 255.0
    return image_as_tensor

def read_cv2_image_as_tensor(cv2_image):
    # Convert from BGR (OpenCV) to RGB
    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    
    # Convert to float32 and normalize to [0, 1]
    rgb_image = rgb_image.astype(np.float32) / 255.0

    # Convert to torch tensor and rearrange to (C, H, W)
    image_as_tensor = torch.from_numpy(rgb_image).permute(2, 0, 1)

    return image_as_tensor

def normalize(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)
    
def apply_ad_scoremap(image, scoremap, alpha=0.5):
    np_image = np.asarray(image, dtype=np.float32)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)

def clip_boxes(boxes, shape):
    if isinstance(boxes, torch.Tensor):  
        boxes[..., 0] = boxes[..., 0].clamp(0, shape[1])  # x1
        boxes[..., 1] = boxes[..., 1].clamp(0, shape[0])  # y1
        boxes[..., 2] = boxes[..., 2].clamp(0, shape[1])  # x2
        boxes[..., 3] = boxes[..., 3].clamp(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2
    return boxes

def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None, padding: bool = True, xywh: bool = False):
    if ratio_pad is None:  
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]) 
        pad = (
            round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1),
            round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1),
        )  
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    if padding:
        boxes[..., 0] -= pad[0]  
        boxes[..., 1] -= pad[1]  # y padding
        if not xywh:
            boxes[..., 2] -= pad[0]  # x padding
            boxes[..., 3] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    return clip_boxes(boxes, img0_shape)

def empty_like(x):
    return (
        torch.empty_like(x, dtype=torch.float32) if isinstance(x, torch.Tensor) else np.empty_like(x, dtype=np.float32)
    )

def xywh2xyxy(x):
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = empty_like(x)  # faster than clone/copy
    xy = x[..., :2]  # centers
    wh = x[..., 2:] / 2  # half width-height
    y[..., :2] = xy - wh  # top left xy
    y[..., 2:] = xy + wh  # bottom right xy
    return y

def non_max_suppression(
    prediction,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    classes=None,
    agnostic: bool = False,
    multi_label: bool = False,
    labels=(),
    max_det: int = 300,
    nc: int = 0,  # number of classes (optional)
    max_time_img: float = 0.05,
    max_nms: int = 30000,
    max_wh: int = 7680,
    in_place: bool = True,
    rotated: bool = False,
    end2end: bool = False,
    return_idxs: bool = False,
    ):

    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output
    if classes is not None:
        classes = torch.tensor(classes, device=prediction.device)

    if prediction.shape[-1] == 6 or end2end:  # end-to-end model (BNC, i.e. 1,300,6)
        output = [pred[pred[:, 4] > conf_thres][:max_det] for pred in prediction]
        if classes is not None:
            output = [pred[(pred[:, 5:6] == classes).any(1)] for pred in output]
        return output

    bs = prediction.shape[0]  # batch size (BCN, i.e. 1,84,6300)
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    nm = prediction.shape[1] - nc - 4  # number of masks
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates
    xinds = torch.stack([torch.arange(len(i), device=prediction.device) for i in xc])[..., None]  # to track idxs

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    time_limit = 2.0 + max_time_img * bs  # seconds to quit after
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)
    if not rotated:
        if in_place:
            prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy
        else:
            prediction = torch.cat((xywh2xyxy(prediction[..., :4]), prediction[..., 4:]), dim=-1)  # xywh to xyxy

    t = time.time()
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    keepi = [torch.zeros((0, 1), device=prediction.device)] * bs  # to store the kept idxs
    for xi, (x, xk) in enumerate(zip(prediction, xinds)):  # image index, (preds, preds indices)
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        filt = xc[xi]  # confidence
        x, xk = x[filt], xk[filt]

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]) and not rotated:
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 4), device=x.device)
            v[:, :4] = xywh2xyxy(lb[:, 1:5])  # box
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = x.split((4, nc, nm), 1)

        if multi_label:
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
            xk = xk[i]
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            filt = conf.view(-1) > conf_thres
            x = torch.cat((box, conf, j.float(), mask), 1)[filt]
            xk = xk[filt]

        # Filter by class
        if classes is not None:
            filt = (x[:, 5:6] == classes).any(1)
            x, xk = x[filt], xk[filt]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        if n > max_nms:  # excess boxes
            filt = x[:, 4].argsort(descending=True)[:max_nms]  # sort by confidence and remove excess boxes
            x, xk = x[filt], xk[filt]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        scores = x[:, 4]  # scores
        if rotated:
            boxes = torch.cat((x[:, :2] + c, x[:, 2:4], x[:, -1:]), dim=-1)  # xywhr
            i = nms_rotated(boxes, scores, iou_thres)
        else:
            boxes = x[:, :4] + c  # boxes (offset by class)
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections

        output[xi], keepi[xi] = x[i], xk[i].reshape(-1)
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return (output, keepi) if return_idxs else output
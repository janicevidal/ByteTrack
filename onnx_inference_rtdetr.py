import argparse
import os

import cv2
import math
import numpy as np
from loguru import logger

import onnxruntime

from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer

from sahi.slicing import slice_image

import random
random.seed(3)

CLASS_COLORS = [[random.randint(0, 255) for _ in range(3)] for _ in range(10)]
CLASS_NAMES = ('pedestrian', 'people', 'bicycle', 'car', 'van', 'truck',
               'tricycle', 'awning-tricycle', 'bus', 'motor')


def make_parser():
    parser = argparse.ArgumentParser("onnxruntime inference sample")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="../../bytetrack_s.onnx",
        help="Input your onnx model.",
    )
    parser.add_argument(
        "-i",
        "--video_path",
        type=str,
        # default='videos/test_uav0000137.mp4',
        default='videos/VID_20230614_174135.mp4',
        help="Path to your input image.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default='demo_output',
        help="Path to your output directory.",
    )
    parser.add_argument(
        "-s",
        "--score_thr",
        type=float,
        default=0.1,
        help="Score threshould to filter the result.",
    )
    parser.add_argument(
        "--input_shape",
        type=str,
        default="608,1088",
        help="Specify an input shape for inference.",
    )

    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.45, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument('--min-box-area', type=float, default=4, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser


class Predictor(object):
    def __init__(self, args):
        self.conf_thres = 0.1
        
        trt_ep_options = {
            "trt_timing_cache_enable": True,
            'trt_fp16_enable': True,
            'trt_layer_norm_fp32_fallback':True
        }
        
        providers=[('TensorrtExecutionProvider', trt_ep_options), 'CUDAExecutionProvider']
        
        self.ort_sess = onnxruntime.InferenceSession(r"E:/PYProject/RT-DETR/rtdetr_paddle/output_visdrone/rtdetr_visdrone.onnx", providers = providers)

        self.args = args
        self.input_shape = tuple(map(int, args.input_shape.split(',')))

    def preprocess(self, image):
        image_h, image_w = image.shape[:2]
        self.ratio_h = 640 / image_h
        self.ratio_w = 640 / image_w
    
        img = cv2.resize(image, (0, 0), fx=self.ratio_w, fy=self.ratio_h, interpolation=1) 
    
        img = img[:, :, ::-1] / 255.0
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img[np.newaxis], dtype=np.float32)
        
        return img, self.ratio_h
    
    def bbox_cxcywh_to_xyxy(self, x):
        bbox = np.zeros_like(x)
        bbox[...,:2] = x[...,:2] - 0.5 * x[...,2:]
        bbox[...,2:] = x[...,:2] + 0.5 * x[...,2:]
        return bbox
    
    def inference(self, ori_img, timer):
        img, ratio = self.preprocess(ori_img)

        timer.tic()
        results = self.ort_sess.run(['sigmoid_19.tmp_0', 'sigmoid_20.tmp_0'], {'image':img})
        
        boxes, scores = [o[0] for o in results]
    
        # 模型后处理
        boxes = self.bbox_cxcywh_to_xyxy(boxes)
        _max = scores.max(-1)
        _mask = _max > self.conf_thres
        boxes, scores = boxes[_mask], scores[_mask]
        boxes = boxes * np.array([640/self.ratio_w, 640/self.ratio_h, 640/self.ratio_w, 640/self.ratio_h], dtype=np.float32)
        labels = scores.argmax(-1)
        scores = scores.max(-1)
        
        return boxes, labels, scores

slice_size = [640, 640]
overlap_ratio = [0.25, 0.25]
combine_method = 'nms'
match_threshold = 0.6
match_metric = 'ios'
num_classes = 11

def multiclass_nms(bboxs, num_classes, match_threshold=0.6, match_metric='iou'):
    final_boxes = []
    for c in range(num_classes):
        idxs = bboxs[:, 0] == c
        if np.count_nonzero(idxs) == 0: continue
        r = nms(bboxs[idxs, 1:], match_threshold, match_metric)
        final_boxes.append(np.concatenate([np.full((r.shape[0], 1), c), r], 1))
    return final_boxes


def nms(dets, match_threshold=0.6, match_metric='iou'):
    """ Apply NMS to avoid detecting too many overlapping bounding boxes.
        Args:
            dets: shape [N, 5], [score, x1, y1, x2, y2]
            match_metric: 'iou' or 'ios'
            match_threshold: overlap thresh for match metric.
    """
    if dets.shape[0] == 0:
        return dets[[], :]
    scores = dets[:, 0]
    x1 = dets[:, 1]
    y1 = dets[:, 2]
    x2 = dets[:, 3]
    y2 = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int32)

    for _i in range(ndets):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        ix1 = x1[i]
        iy1 = y1[i]
        ix2 = x2[i]
        iy2 = y2[i]
        iarea = areas[i]
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            xx1 = max(ix1, x1[j])
            yy1 = max(iy1, y1[j])
            xx2 = min(ix2, x2[j])
            yy2 = min(iy2, y2[j])
            w = max(0.0, xx2 - xx1 + 1)
            h = max(0.0, yy2 - yy1 + 1)
            inter = w * h
            if match_metric == 'iou':
                union = iarea + areas[j] - inter
                match_value = inter / union
            elif match_metric == 'ios':
                smaller = min(iarea, areas[j])
                match_value = inter / smaller
            else:
                raise ValueError()
            if match_value >= match_threshold:
                suppressed[j] = 1
    keep = np.where(suppressed == 0)[0]
    dets = dets[keep, :]
    return dets


def imageflow_demo(predictor, args):
    cap = cv2.VideoCapture(args.video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    save_folder = args.output_dir
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, args.video_path.split("/")[-1])
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    tracker = BYTETracker(args, frame_rate=25)
    timer = Timer()
    frame_id = 0
    results = []
    while True:
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        ret_val, frame = cap.read()
        if ret_val: 
            slice_image_result = slice_image(
                image=frame,
                slice_height=slice_size[0],
                slice_width=slice_size[1],
                overlap_height_ratio=overlap_ratio[0],
                overlap_width_ratio=overlap_ratio[1])
    
            sub_img_num = len(slice_image_result)
            merged_bboxs = []
            # print('slice to {} sub_samples.', sub_img_num)
    
            batch_image_list = [
                    slice_image_result.images[_ind] for _ind in range(sub_img_num)
                ]
            
            all_bboxs = []
            bboxs_num = []
    
            img_info = {"id": 0}
            height, width = frame.shape[:2]
            img_info["height"] = height
            img_info["width"] = width
            img_info["raw_img"] = frame
        
            for img in batch_image_list:
                im_c, im_h, im_w = img.shape[:]
                boxes, labels, scores = predictor.inference(img, timer)
        
                count = 0
        
                for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                    x1, y1, x2, y2 = box
                    box_array = np.array([label, score, x1, y1, x2, y2])
            
                    all_bboxs.append(box_array)
                
                    count = count + 1
        
                np_boxes_num = np.array([count])
        
                bboxs_num.append(np_boxes_num)
    
    
            result = dict(boxes=all_bboxs, boxes_num=bboxs_num)
            
            st, ed = 0, result['boxes_num'][0][0]  # start_index, end_index
    
            for _ind in range(sub_img_num):
                boxes_num = result['boxes_num'][_ind][0]
                ed = st + boxes_num
                shift_amount = slice_image_result.starting_pixels[_ind]
        
                result_box_array = np.array(result['boxes'][st:ed])
        
                if len(result_box_array) == 0:
                    continue

                result_box_array[:, 2:4] = result_box_array[:, 2:4] + np.array(shift_amount)
                result_box_array[:, 4:6] = result_box_array[:, 4:6] + np.array(shift_amount)

                merged_bboxs.append(result_box_array)
            
                st = ed
        
            merged_results = {'boxes': []}
            if combine_method == 'nms':
                final_boxes = multiclass_nms(np.concatenate(merged_bboxs), num_classes, match_threshold, match_metric)
                merged_results['boxes'] = np.concatenate(final_boxes)
            elif combine_method == 'concat':
                    merged_results['boxes'] = np.concatenate(merged_bboxs)
            else:
                raise ValueError(
                    "Now only support 'nms' or 'concat' to fuse detection results."
                )
        
            merged_results['boxes_num'] = np.array([len(merged_results['boxes'])], dtype=np.int32)
            
            # for dt in merged_results['boxes']:
            #     clsid, bbox, score = int(dt[0]), dt[2:], dt[1]
        
            #     xmin, ymin, xmax, ymax = bbox
            #     xmin = math.floor(min(max(1, xmin), img_info["width"] - 1))
            #     ymin = math.floor(min(max(1, ymin), img_info["height"] - 1))
            #     xmax = math.ceil(min(max(1, xmax), img_info["width"] - 1))
            #     ymax = math.ceil(min(max(1, ymax), img_info["height"] - 1))
            #     cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), CLASS_COLORS[clsid], 2)
                
            # vid_writer.write(frame)
    
            online_targets = tracker.update(merged_results['boxes'], [img_info['height'], img_info['width']], [img_info['height'], img_info['width']])
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                # vertical = tlwh[2] / tlwh[3] > 1.6
                # if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                if tlwh[2] * tlwh[3] > args.min_box_area:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
            timer.toc()
            results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))
            online_im = plot_tracking(img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1,
                                      fps=1. / timer.average_time)
            vid_writer.write(online_im)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        frame_id += 1


if __name__ == '__main__':
    args = make_parser().parse_args()

    predictor = Predictor(args)
    imageflow_demo(predictor, args)

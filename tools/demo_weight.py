# get frames by a file (i.e. mp4, avi ... ) or RTSP
# yolox-nano predicts bboxes in both images
# filters images which has sufficient size and appropriate position(center) and also depth (checkerboard)
# matches the bboxes
# estimate the depth
# predicts the length
# predicts the weight
# visualize the weight

#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import time
from loguru import logger

import cv2

import torch

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import SALMON_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis
import numpy as np

from scipy.optimize import linear_sum_assignment

import sys
sys.path.insert(0, '../mmpose')
from mmpose.apis import init_pose_model, inference_top_down_pose_model
from mmpose.datasets import DatasetInfo
import copy
IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")
    parser.add_argument(
        "--demo", default="video", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("--pose_config",
                        type=str,
                        default='../mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/salmon/mobilenetv2_salmon_256x192.py')
    parser.add_argument("--pose_checkpoint",
                        type=str,
                        default='../mmpose/work_dirs/mobilenetv2_salmon_256x192/epoch_6040.pth')

    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "--path", default="C:/Records/Local Records/Ch1_169.254.152.185/20221227151222090.avi", help="path to images or video"
    )
    parser.add_argument(
        "--path2", default="C:/Records/Local Records/Ch1_169.254.152.184/20221227151220082.avi", help="path to images or video 2"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default='exps/example/custom/yolox_nano_salmon.py',
        type=str,
        help="please input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default='weights/best_ckpt.pth', type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.5, type=float, help="test conf")
    parser.add_argument("--nms", default=0.45, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names

def vis(img, boxes, scores, cls_ids, weights, conf=0.5):

    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (np.array([0.000, 0.447, 0.741]) * 255).astype(np.uint8).tolist()
        text = 'weight: {:.2f} g'.format(weights[i])
        txt_color = (0, 0, 0) if np.mean(np.array([0.000, 0.447, 0.741])) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 2, 1)[0] # * size has been multiplied by 5
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 20) #

        txt_bk_color = (np.array([0.000, 0.447, 0.741]) * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 2, txt_color, thickness=5) #

    return img

class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=SALMON_CLASSES,
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def visual(self, output, img_info, weights, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, weights, cls_conf)
        return vis_res


def image_demo(predictor, vis_folder, path, current_time, save_result):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    for image_name in files:
        outputs, img_info = predictor.inference(image_name)
        result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        if save_result:
            save_folder = os.path.join(
                vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            )
            os.makedirs(save_folder, exist_ok=True)
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            logger.info("Saving detection result in {}".format(save_file_name))
            cv2.imwrite(save_file_name, result_image)
        ch = cv2.waitKey()
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break

def match_bbox(outputs, outputs2):
    """
    Args:
        outputs: [ndarray (n,7)]
        outputs2: [ndarray (m,7)]
    Returns:

    """
    outputs = outputs[0]
    outputs2 = outputs2[0]
    # always same sahpe of outputs[0] and outputs2[0]? -> NO
    # cost matrix should be the shape of (min(n,m),min(n,m))
    if outputs==None or outputs2==None:
        return None, None, None
    cost = lambda b1,b2: torch.abs(b1[[1,3]]-b2[[1,3]]).sum()
    min_cost = np.inf
    # row, col = None, None
    cost_matrix = np.zeros((outputs.shape[0], outputs2.shape[0]))
    for i in range(cost_matrix.shape[0]):
        for j in range(cost_matrix.shape[1]):
            cost_matrix[i][j] = cost(outputs[i], outputs2[j])
    rows, cols = linear_sum_assignment(cost_matrix)
    costs = [cost_matrix[row][col] for row,col in zip(rows,cols)]

    # for i in range(outputs.shape[0]):
    #     for j in range(outputs2.shape[0]):
    #         if min_cost > cost(outputs[i], outputs2[j]):
    #             min_cost = cost(outputs[i], outputs2[j])
    #             row, col = i, j
    return rows, cols, costs

def get_depth(Left_Stereo_Map, Right_Stereo_Map, point, point2):
    # j1, i1 = int(point[0].item()),int(point[1].item())
    # j2, i2 = int(point2[0].item()),int(point2[1].item())
    # rectified_point = Left_Stereo_Map[0][i1, j1] # The map has i j order but each elements are x, y order
    # rectified_point2 = Right_Stereo_Map[0][i2, j2]
    rectified_point = point
    rectified_point2 = point2
    print(point, point2)
    print(rectified_point, rectified_point2)
    disparity = rectified_point[0] - rectified_point2[0]
    baseline = 0.09
    focal_length = 2.15309798e+03
    depth = baseline * focal_length / disparity
    return depth

def get_length(Stereo_Map, x1, y1, x2, y2, depth1, depth2):
    fx = 2.15309798e+03
    fy = 2.16339297e+03
    cx = 1.29010772e+03
    cy = 1.00323450e+03
    # j1, i1 = int(x1.item()), int(y1.item())
    # j2, i2 = int(x2.item()), int(y2.item())
    # j1 = min(Stereo_Map[0].shape[1] - 1, j1)
    # i1 = min(Stereo_Map[0].shape[0] - 1, i1)
    # j2 = min(Stereo_Map[0].shape[1] - 1, j2)
    # i2 = min(Stereo_Map[0].shape[0] - 1, i2)
    # x1, y1 = Stereo_Map[0][i1, j1] # rectified
    # x2, y2 = Stereo_Map[0][i2, j2]
    real_x1 = (x1-cx) / fx * depth1
    real_y1 = (y1-cy) / fy * depth1
    real_x2 = (x2-cx) / fx * depth2
    real_y2 = (y2-cx) / fy * depth2
    lx = real_x1 - real_x2
    ly = real_y1 - real_y2
    lz = depth1 - depth2
    length = np.sqrt(lx ** 2 + ly ** 2 + lz ** 2)
    length *= 1000 # unit: mm
    return length

def get_weight(length):
    k = 1.3
    weight = k * (length ** 3) / (10 ** 5)
    return weight

def vis_keypoints(frame, keypoints):
    """
    Args:
        frame: RGB with size of 1944 2592
        keypoints: [{'bbox':..., 'keypoints':numpy.ndarray(12,3)}]
    Returns:
        frame where every keypoints are visualized.
    """
    # for keypoint in keypoints:
    #     x,y,c = keypoint
    #     if c >= 0.1:
    #         frame = cv2.circle(frame, (int(x),int(y)), radius=10, thickness=-1, color=(255,0,0))
    frame = cv2.circle(frame, (int(keypoints[1,0]),int(keypoints[1,1])), radius=10, thickness=-1, color=(0,0,255))
    frame = cv2.circle(frame, (int(keypoints[9,0]),int(keypoints[9,1])), radius=10, thickness=-1, color=(255,0,0))

    return frame

def get_weight_from_head_tail_coords(Left_Stereo_Map,Right_Stereo_Map,head1_x, head1_y,tail1_x, tail1_y,head2_x, head2_y,tail2_x, tail2_y):
    head_depth = get_depth(Left_Stereo_Map, Right_Stereo_Map, (head1_x, head1_y), (head2_x, head2_y))
    tail_depth = get_depth(Left_Stereo_Map, Right_Stereo_Map, (tail1_x, tail1_y), (tail2_x, tail2_y))
    length = get_length(Left_Stereo_Map, head1_x, head1_y, tail1_x, tail1_y, head_depth, tail_depth)
    weight = get_weight(length)
    print('head_depth: {:2f}, tail_depth: {:.2f}, length: {:.2f} mm, weight: {:.2f} g'.format(head_depth, tail_depth, length, weight))
    return weight


def imageflow_demo(predictor, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    cap2 = cv2.VideoCapture(args.path2 if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)

    width2 = cap2.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height2 = cap2.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps2 = cap2.get(cv2.CAP_PROP_FPS)

    assert width == width2 and height == height2 and fps == fps2, 'Two videos are not correspondent.'
    if args.save_result:
        save_folder = os.path.join(
            vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        )
        os.makedirs(save_folder, exist_ok=True)
        if args.demo == "video":
            save_path = os.path.join(save_folder, os.path.basename(args.path))
        else:
            save_path = os.path.join(save_folder, "camera.mp4")
        logger.info(f"video save_path is {save_path}")
        vid_writer = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
        )
    lag = 47 # TODO: do not hard code. make them function
    cap.set(cv2.CAP_PROP_POS_FRAMES, 437)
    cap2.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + lag)
    cv_file = cv2.FileStorage("../StereoCalibration/params_230103.xml", cv2.FILE_STORAGE_READ)
    Left_Stereo_Map = cv_file.getNode("Left_Stereo_Map_x").mat(), cv_file.getNode("Left_Stereo_Map_y").mat()
    Right_Stereo_Map = cv_file.getNode("Right_Stereo_Map_x").mat(), cv_file.getNode("Right_Stereo_Map_y").mat()
    cv_file.release()

    pose_model = init_pose_model(args.pose_config, args.pose_checkpoint, device='cuda' if args.device=='gpu' else 'cpu')
    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    dataset_info = DatasetInfo(dataset_info)
    return_heatmap = False
    output_layer_names = None

    while True:
        ret_val, frame = cap.read()
        ret_val2, frame2 = cap2.read()
        # Resize the source image for visualization. This is not related to preprocessing
        # frame = cv2.resize(frame, None, fx=0.25, fy=0.25)
        # frame2 = cv2.resize(frame2, None, fx=0.25, fy=0.25)
        frame = cv2.remap(frame, Left_Stereo_Map[0], Left_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        frame2 = cv2.remap(frame2, Right_Stereo_Map[0], Right_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        if ret_val and ret_val2:
            outputs, img_info = predictor.inference(frame) # it returns list with length of 1 is it batchsize?
            outputs2, img_info2 = predictor.inference(frame2)
            outputs = outputs[0]
            outputs2 = outputs2[0]
            """
            if outputs[0] != None and outputs2[0] != None:
                rows, cols, costs = match_bbox(outputs, outputs2)
                outputs = outputs[0][rows, :]
                outputs2 = outputs2[0][cols, :]
                # score = (outputs[0][0,4]*outputs[0][0,5]).item()
                ratio = img_info["ratio"]
                # cx1 = outputs[0][0][[0, 2]].sum() / 2 / ratio
                # cy1 = outputs[0][0][[1, 3]].sum() / 2 / ratio
                # cx2 = outputs2[0][0][[0, 2]].sum() / 2 / ratio
                # cy2 = outputs2[0][0][[1, 3]].sum() / 2 / ratio
                # h = (outputs[0][0][3] - outputs[0][0][1]).item() / ratio
                # w = (outputs[0][0][2] - outputs[0][0][0]).item() / ratio
                # center = (cx1, cy1)/
                # center2 = (cx2, cy2)

                # x1,y1,x2,y2,_,_,_ = outputs[0][0] / ratio
                # bbox_x_len = (outputs[0][0][2] - outputs[0][0][0]).item()
                # print('bounding box x length: {}'.format(bbox_x_len))
                # if cx1 <= cx2 or h > w:# or h < frame.shape[0] * 0.2 or w < frame.shape[1] * 0.2:
                #     outputs = [None]
                #     outputs2 = [None]
                #     result_frame_both = np.hstack((frame, frame2))
                # depth = get_depth(Left_Stereo_Map,Right_Stereo_Map,center, center2)
                # length = get_length(Left_Stereo_Map, x1, y1, x2, y2, depth)
                # weight = get_weight(length)

                # Keypoint estimation
                # bboxes = []
                # for row in rows:
                #     bboxes.append(outputs[row, :4].cpu().numpy() / ratio)
                bboxes = copy.deepcopy(outputs[:, :4].cpu().numpy() / ratio)
                person_results = [{'bbox': bbox} for bbox in bboxes]
                pose_results, returned_outputs = inference_top_down_pose_model(
                    pose_model,
                    frame,
                    person_results,
                    format='xyxy',
                    dataset=dataset,
                    dataset_info=dataset_info,
                    return_heatmap=return_heatmap,
                    outputs=output_layer_names
                )
                bboxes2 = copy.deepcopy(outputs2[:, :4].cpu().numpy() / ratio)
                person_results2 = [{'bbox': bbox} for bbox in bboxes2]
                pose_results2, returned_outputs2 = inference_top_down_pose_model(
                    pose_model,
                    frame2,
                    person_results2,
                    format='xyxy',
                    dataset=dataset,
                    dataset_info=dataset_info,
                    return_heatmap=return_heatmap,
                    outputs=output_layer_names
                )

                # visualize keypoints
                points_all = False
                weight_predicted_indices = []
                weights = []
                conf_thre = 0.0
                for i, (pose_result, pose_result2) in enumerate(zip(pose_results, pose_results2)):
                    keypoints = pose_result['keypoints']
                    keypoints2 = pose_result2['keypoints']
                    head1_c = keypoints[1, 2]
                    tail1_c = keypoints[9, 2]
                    head2_c = keypoints2[1, 2]
                    tail2_c = keypoints2[9, 2]
                    if head1_c >= conf_thre and tail1_c >= conf_thre and head2_c >= conf_thre and tail2_c >= conf_thre:
                        # calcul the depth and weight
                        head1_x, head1_y = keypoints[1, :2]
                        tail1_x, tail1_y = keypoints[9, :2]
                        head2_x, head2_y = keypoints2[1, :2]
                        tail2_x, tail2_y = keypoints2[9, :2]

                        weight = get_weight_from_head_tail_coords(Left_Stereo_Map,
                                                                  Right_Stereo_Map,
                                                                  head1_x, head1_y,
                                                                  tail1_x, tail1_y,
                                                                  head2_x, head2_y,
                                                                  tail2_x, tail2_y)
                        weights.append(weight)
                        img_info['raw_img'] = vis_keypoints(img_info['raw_img'], keypoints)
                        img_info2['raw_img'] = vis_keypoints(img_info2['raw_img'], keypoints2)
                        points_all = True
                        weight_predicted_indices.append(i)
                outputs = outputs[weight_predicted_indices, :]
                outputs2 = outputs2[weight_predicted_indices, :]
                # visualize bounding box
                weight = 0.0 # temporary
                result_frame = predictor.visual(outputs, img_info, weights, predictor.confthre)
                result_frame2 = predictor.visual(outputs2, img_info2, weights, predictor.confthre)
                result_frame_both = np.hstack((result_frame, result_frame2))
            else:
                outputs = [None]
                outputs2 = [None]
                result_frame_both = np.hstack((frame, frame2))
            """
            outputs = outputs / img_info['ratio']
            outputs2 = outputs2 / img_info2['ratio']
            outputs = outputs[0:1, :]
            outputs2 = outputs2[0:1, :]
            print(outputs)
            weights = [0.0] * outputs.shape[0]
            weights2 = [0.0] * outputs2.shape[0]
            result_frame = predictor.visual(outputs, img_info, weights, predictor.confthre)
            result_frame2 = predictor.visual(outputs2, img_info2, weights2, predictor.confthre)
            result_frame_both = np.hstack((result_frame, result_frame2))
            if args.save_result:
                vid_writer.write(result_frame_both)
            else:
                cv2.namedWindow("yolox", cv2.WINDOW_NORMAL)
                cv2.imshow("yolox", result_frame_both)
            # if points_all:
            #     print(cap.get(cv2.CAP_PROP_POS_FRAMES))
            #     ch = cv2.waitKey()
            # else:
            ch = cv2.waitKey()
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    vis_folder = None
    if args.save_result:
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()  # to FP16
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(
        model, exp, SALMON_CLASSES, trt_file, decoder,
        args.device, args.fp16, args.legacy,
    )
    current_time = time.localtime()
    if args.demo == "image":
        image_demo(predictor, vis_folder, args.path, current_time, args.save_result)
    elif args.demo == "video" or args.demo == "webcam":
        imageflow_demo(predictor, vis_folder, current_time, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)

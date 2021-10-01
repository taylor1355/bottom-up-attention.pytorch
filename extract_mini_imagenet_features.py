# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# pylint: disable=no-member
"""
TridentNet Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
import argparse
import os
import sys
import torch
import pdb
# import tqdm
import cv2
import numpy as np
import pickle
sys.path.append('detectron2')

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_setup, launch
from detectron2.evaluation import COCOEvaluator, verify_results
from detectron2.structures import Instances

from utils.utils import mkdir, save_features
from utils.extract_utils import get_image_blob, save_bbox, save_roi_features_by_bbox, save_roi_features
from utils.progress_bar import ProgressBar
from models import add_config
from models.bua.box_regression import BUABoxes
from models.bua.layers.nms import nms

import ray
from ray.actor import ActorHandle

def read_pickle(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle, encoding='latin1')

def read_label_mapper(path):
    d = {}
    with open(path, 'r') as handle:
        for line in handle:
            code, labels = line.strip().split(': ')
            d[code] = labels.strip().split(',')
    return d

def switch_extract_mode(mode, add_proposal_generator=False):
    if mode == 'roi_feats':
        switch_cmd = ['MODEL.BUA.EXTRACTOR.MODE', 1]
    elif mode == 'bboxes':
        switch_cmd = ['MODEL.BUA.EXTRACTOR.MODE', 2]
    elif mode == 'bbox_feats':
        switch_cmd = ['MODEL.BUA.EXTRACTOR.MODE', 3, 'MODEL.PROPOSAL_GENERATOR.NAME', 'PrecomputedProposals']
    elif mode == 'all_in_one':
        switch_cmd = ['MODEL.BUA.EXTRACTOR.MODE', 4]
        if add_proposal_generator:
            switch_cmd += ['MODEL.PROPOSAL_GENERATOR.NAME', 'PrecomputedProposals']
    else:
        print('Wrong extract mode! ')
        exit()
    return switch_cmd

def set_min_max_boxes(min_max_boxes):
    if min_max_boxes == 'min_max_default':
        return []
    try:
        min_boxes = int(min_max_boxes.split(',')[0])
        max_boxes = int(min_max_boxes.split(',')[1])
    except:
        print('Illegal min-max boxes setting, using config default. ')
        return []
    cmd = ['MODEL.BUA.EXTRACTOR.MIN_BOXES', min_boxes, 
            'MODEL.BUA.EXTRACTOR.MAX_BOXES', max_boxes]
    return cmd

def setup(args, add_proposal_generator=False):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_config(args, cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.merge_from_list(switch_extract_mode(args.extract_mode, add_proposal_generator=add_proposal_generator))
    cfg.merge_from_list(set_min_max_boxes(args.min_max_boxes))
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def generate_npz(extract_mode, *args):
    if extract_mode == 1:
        save_roi_features(*args)
    elif extract_mode == 2:
        save_bbox(*args)
    elif extract_mode == 3:
        save_roi_features_by_bbox(*args)
    else:
        print('Invalid Extract Mode! ')

def prune_extractions(cfg, dataset_dict, boxes_list, scores_list):
    # declare maximum retention boxes
    MAX_BOXES = cfg.MODEL.BUA.EXTRACTOR.MAX_BOXES
    # We assume scores and boxes is a list/tuple of a single item
    scores = scores_list[0]
    boxes = boxes_list[0]
    # scores = [Num_Detections x (Num_Classes + 1)]. I believe the 0th index
    # is a "null" term.
    num_classes = scores.shape[1]
    # Clip boxes according to the actual image size
    boxes = BUABoxes(boxes.reshape(-1, 4))
    boxes.clip(
        (
            dataset_dict['image'].shape[1]/dataset_dict['im_scale'], 
            dataset_dict['image'].shape[2]/dataset_dict['im_scale']
        )
    )
    # Reshape boxes to be [Num_Detections x Num_Classes*4]
    # The idea is that we have a candidate box detection for each class
    boxes = boxes.tensor.view(-1, num_classes*4) 
    # Iterate through each detection, and select box whose class has 
    # highest confidence
    cls_boxes = torch.zeros((boxes.shape[0], 4))
    for idx in range(boxes.shape[0]):
        cls_idx = torch.argmax(scores[idx, 1:]) + 1
        cls_boxes[idx, :] = boxes[idx, cls_idx * 4: (cls_idx + 1) * 4]
    # Now we have cls_boxes: [Num_Classes x 4]. We need to see how
    # valuable each detection is. We iterate through each class and
    # apply NMS on boxes from each class.
    max_conf = torch.zeros((scores.shape[0])).to(scores.device)
    for cls_ind in range(1, num_classes):
            cls_scores = scores[:, cls_ind]
            keep = nms(cls_boxes, cls_scores, 0.3)
            max_conf[keep] = torch.where(
                cls_scores[keep] > max_conf[keep],
                cls_scores[keep],
                max_conf[keep]
            )
    # Keep top-K most informative detections according to the class scores
    keep_boxes = torch.argsort(max_conf, descending=True)[:MAX_BOXES]
    image_bboxes = cls_boxes[keep_boxes]

    return image_bboxes, keep_boxes

# @ray.remote(num_gpus=1)
def extract_feat(split_idx, img_list, cfg, args, actor: ActorHandle, root_idx):
    num_images = len(img_list)
    print('Number of images on split{}: {}.'.format(split_idx, num_images))

    model = DefaultTrainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )
    model.eval()

    for idx, im_file in enumerate(img_list):
        # pdb.set_trace()
        im = im_file#.transpose(2, 0, 1)
        # im = cv2.imread(os.path.join(args.image_dir, im_file))
        # if im is None:
        #     print(os.path.join(args.image_dir, im_file), "is illegal!")
        #     actor.update.remote(1)
        #     continue
        dataset_dict = get_image_blob(im, cfg.MODEL.PIXEL_MEAN)
        
        # pdb.set_trace()
        cfg = setup(args, add_proposal_generator=False)
        model = DefaultTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        model.eval()
        with torch.set_grad_enabled(False):
            _, boxes, scores, old_feats, old_attrs = model([dataset_dict])
        boxes = [box.cpu() for box in boxes]
        scores = [score.cpu() for score in scores]
        
        pruned_bboxes, keep_indices = prune_extractions(
            cfg, dataset_dict, boxes, scores
            )

        cfg = setup(args, add_proposal_generator=True)
        model = DefaultTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        model.eval()
        
        bbox = pruned_bboxes * dataset_dict['im_scale']
        proposals = Instances(dataset_dict['image'].shape[-2:])
        proposals.proposal_boxes = BUABoxes(bbox)
        dataset_dict['proposals'] = proposals
        # pdb.set_trace()
        attr_scores = None
        with torch.set_grad_enabled(False):
            if cfg.MODEL.BUA.ATTRIBUTE_ON:
                new_boxes, _, new_scores, features_pooled, attr_scores = model([dataset_dict])
            else:
                boxes, _, scores, features_pooled = model([dataset_dict])
        # pdb.set_trace()
        new_boxes = [box.tensor.cpu() for box in new_boxes]
        new_scores = [score.cpu() for score in new_scores]
        features_pooled = [feat.cpu() for feat in features_pooled]
        if not attr_scores is None:
            attr_scores = [attr_score.data.cpu() for attr_score in attr_scores]
        # pdb.set_trace()
        generate_npz(3, 
            args, cfg, root_idx + idx, im, dataset_dict, 
            new_boxes, new_scores, features_pooled, attr_scores)
            


        actor.update.remote(1)


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection2 Inference")
    parser.add_argument(
        "--config-file",
        default="configs/bua-caffe/extract-bua-caffe-r101.yaml",
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument('--num-cpus', default=1, type=int, 
                        help='number of cpus to use for ray, 0 means no limit')

    parser.add_argument('--gpus', dest='gpu_id', help='GPU id(s) to use',
                        default='0', type=str)

    parser.add_argument("--mode", default="caffe", type=str, help="bua_caffe, ...")

    parser.add_argument('--extract-mode', default='roi_feats', type=str,
                        help="'roi_feats', 'bboxes' and 'bbox_feats' indicates \
                        'extract roi features directly', 'extract bboxes only' and \
                        'extract roi features with pre-computed bboxes' respectively")

    parser.add_argument('--min-max-boxes', default='min_max_default', type=str, 
                        help='the number of min-max boxes of extractor')

    parser.add_argument('--out-dir', dest='output_dir',
                        help='output directory for features',
                        default="features")
    parser.add_argument('--input-dir', dest='input_dir',
                        help='directory with images',
                        default="image")
    parser.add_argument('--bbox-dir', dest='bbox_dir',
                        help='directory with bbox',
                        default="bbox")
    parser.add_argument('--keep-k', dest='keep_k', default=10)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="whether to attempt to resume from the checkpoint directory",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    cfg = setup(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    num_gpus = len(args.gpu_id.split(','))

    MIN_BOXES = cfg.MODEL.BUA.EXTRACTOR.MIN_BOXES
    MAX_BOXES = cfg.MODEL.BUA.EXTRACTOR.MAX_BOXES
    CONF_THRESH = cfg.MODEL.BUA.EXTRACTOR.CONF_THRESH

    # Specify File Names
    data_dir = vars(args)['input_dir']
    train_file = os.path.join(data_dir, 'miniImageNet_category_split_train_phase_train.pickle')
    # train_file = os.path.join(data_dir, 'vg_pickle.pickle')
    dev_file = os.path.join(data_dir, 'miniImageNet_category_split_val.pickle')
    test_file = os.path.join(data_dir, 'miniImageNet_category_split_test.pickle')
    map_file = os.path.join(data_dir, 'label_code2name.txt')
    # Read Data
    train_data = read_pickle(train_file)
    code2label = read_label_mapper(map_file)
    keep_k = vars(args)['keep_k']

    class2images = {}
    for image, label in zip(train_data['data'], train_data['labels']):
        if label not in class2images:
            class2images[label] = []
        if len(class2images[label]) < keep_k: 
            class2images[label].append(image)
    # pdb.set_trace()
    num_images = len(train_data['data'][:keep_k])
    print('Number of images: {}.'.format(num_images))
    output_dir = args.output_dir
    for class_idx, images in class2images.items():
        img_lists = [images[i::num_gpus] for i in range(num_gpus)]
        # pdb.set_trace()
        pb = ProgressBar(num_images)
        actor = pb.actor

        print('Number of GPUs: {}.'.format(num_gpus))
        extract_feat_list = []
        root_idx = 0
        args.output_dir = os.path.join(output_dir, str(class_idx))
        os.makedirs(args.output_dir, exist_ok=True)
        for i in range(num_gpus):
            extract_feat_list.append(extract_feat(i, img_lists[i], cfg, args, actor, root_idx))
            root_idx += len(img_lists[i])
        
        pb.print_until_done()

if __name__ == "__main__":
    main()

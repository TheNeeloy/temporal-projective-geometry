# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import multiprocessing as mp
import os
import cv2
from tqdm import tqdm
import numpy as np
import json

from predictor import VisualizationDemo
from adet.config import get_cfg
import torch


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.MEInst.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.MODEL.WEIGHTS = "/home/neeloy/projects/cs445/final_project/SSIS/tools/output/SSISv2_MS_R_101_bifpn_with_offset_class_maskiouv2_da_bl/model_ssisv2_final.pth"
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument(
        "--config-file",
        default="/home/neeloy/projects/cs445/final_project/SSIS/configs/SSIS/MS_R_101_BiFPN_SSISv2_demo.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input", 
        default="/home/neeloy/projects/cs445/final_project/temporal-projective-geometry/data/fake_video_paths.txt", 
        help="Path to txt file with paths to videos in dataset"
    )
    parser.add_argument(
        "--raw_frames", 
        default="/home/neeloy/projects/cs445/final_project/temporal-projective-geometry/data/raw_frames/fake", 
        help="Path to folder with raw frames for videos in input txt file"
    )
    parser.add_argument(
        "--output",
        default="/home/neeloy/projects/cs445/final_project/temporal-projective-geometry/data/object_shadow_masks/fake",
        help="Parent directory of where to save masks for each video",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.1,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":

    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg)
    f = open(args.input, "r")

    # Iterate over video paths
    for line in tqdm(f):

        # Check if raw frames folder exists for current video
        vid_path = os.path.join(line.strip())
        vid_file_name = (os.path.split(vid_path)[1]).split(".")[0]
        vid_raw_frames_dir = os.path.join(args.raw_frames, vid_file_name)
        if not os.path.exists(vid_raw_frames_dir):
            raise Exception

        # Make save directories for video
        vid_save_dir = os.path.join(args.output, vid_file_name)
        object_save_dir = os.path.join(vid_save_dir, "object_masks")
        shadow_save_dir = os.path.join(vid_save_dir, "shadow_masks")
        if not os.path.exists(vid_save_dir):
            os.makedirs(vid_save_dir)
            os.makedirs(object_save_dir)
            os.makedirs(shadow_save_dir)

        # Load video frames and predict masks
        valid_mask_frame_ids = {"valid_ids":[]}
        frame_id = 0

        while(True): 

            curr_frame_path = os.path.join(vid_raw_frames_dir, "{}.png".format(frame_id))
            if not os.path.isfile(curr_frame_path):
                break
            curr_frame = cv2.imread(curr_frame_path)  # (H, W, 3)

            torch.cuda.empty_cache()
            with torch.no_grad():
                instances, visualized_output = demo.run_on_image(curr_frame)

            # instances.pred_classes is a list of length num_predictions*2 with values 0 or 1
            # instances.pred_masks is a numpy array of shape (num_predictions*2, H, W)

            # Combine predicted masks together
            shadows_mask = np.zeros((instances.pred_masks.shape[1:]))   # (H, W)
            objects_mask = np.zeros((instances.pred_masks.shape[1:]))   # (H, W)
            for pred_mask, pred_class in zip(instances.pred_masks, instances.pred_classes):
                if pred_class:
                    shadows_mask = np.logical_or(shadows_mask, pred_mask)
                else:
                    objects_mask = np.logical_or(objects_mask, pred_mask)

            # Image has valid predictions
            if len(instances) > 0:

                valid_mask_frame_ids["valid_ids"].append(frame_id)

                # Save masks
                shadow_save_path = os.path.join(shadow_save_dir, "{}.png".format(frame_id))
                cv2.imwrite(shadow_save_path, shadows_mask.astype(np.uint8)*255)
                object_save_path = os.path.join(object_save_dir, "{}.png".format(frame_id))
                cv2.imwrite(object_save_path, objects_mask.astype(np.uint8)*255)
                break

                # Uncomment to save combined masks on original image
                # visualized_output.save(os.path.join(vid_save_dir, "{}_visualized.png".format(frame_id)))

            frame_id += 1

        # Convert and write JSON object to file
        with open(os.path.join(vid_save_dir, "valid_frames.json"), "w") as outfile: 
            json.dump(valid_mask_frame_ids, outfile, indent=4)

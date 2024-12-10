# Standard Imports
import argparse
import os

# Third Party Imports
import cv2
from tqdm import tqdm
import torch
from perspective2d import PerspectiveFields


def get_parser():
    parser = argparse.ArgumentParser(description="Perspective Fields Extraction Script")
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
        default="/home/neeloy/projects/cs445/final_project/temporal-projective-geometry/data/perspective_fields/fake",
        help="Parent directory of where to save fields for each video",
    )
    return parser

if __name__ == "__main__":

    # Parse args
    args = get_parser().parse_args()
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    f = open(args.input, "r")

    # Load model
    version = 'Paramnet-360Cities-edina-centered'
    # version = 'Paramnet-360Cities-edina-uncentered'
    # version = 'PersNet_Paramnet-GSV-centered'
    # version = 'PersNet_Paramnet-GSV-uncentered'
    # version = 'PersNet-360Cities'
    pf_model = PerspectiveFields(version).eval().cuda()

    # Iterate over video paths
    for line_num, line in enumerate(tqdm(f)):

        # Check if raw frames folder exists for current video
        vid_path = os.path.join(line.strip())
        vid_file_name = (os.path.split(vid_path)[1]).split(".")[0]
        vid_raw_frames_dir = os.path.join(args.raw_frames, vid_file_name)
        if not os.path.exists(vid_raw_frames_dir):
            raise Exception

        # Make save directory for video
        vid_save_dir = os.path.join(args.output, vid_file_name)
        if not os.path.exists(vid_save_dir):
            os.makedirs(vid_save_dir)

        # Load video frames and predict fields
        frame_id = 0

        while(True): 

            curr_frame_path = os.path.join(vid_raw_frames_dir, "{}.png".format(frame_id))
            if not os.path.isfile(curr_frame_path):
                break
            curr_frame = cv2.imread(curr_frame_path)  # (H, W, 3)

            predictions = pf_model.inference(img_bgr=curr_frame)
            predictions["pred_gravity_original"] = predictions["pred_gravity_original"].cpu().detach()
            predictions["pred_latitude_original"] = predictions["pred_latitude_original"].cpu().detach()

            torch.save(predictions, os.path.join(vid_save_dir, "{}.pt".format(frame_id)))

            frame_id += 1

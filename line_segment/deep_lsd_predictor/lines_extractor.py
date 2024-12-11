# Standard Imports
import argparse
import os
import pickle

# Third Party Imports
import cv2
from tqdm import tqdm
import torch
from deeplsd.models.deeplsd_inference import DeepLSD


def get_parser():
    parser = argparse.ArgumentParser(description="Line Segment Extraction Script")
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
        default="/home/neeloy/projects/cs445/final_project/temporal-projective-geometry/data/line_segments/fake",
        help="Parent directory of where to save line segments for each video",
    )
    return parser

if __name__ == "__main__":

    # Parse args
    args = get_parser().parse_args()
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    f = open(args.input, "r")

    # Model config
    conf = {
        'detect_lines': True,  # Whether to detect lines or only DF/AF
        'line_detection_params': {
            'merge': False,  # Whether to merge close-by lines
            'filtering': True,  # Whether to filter out lines based on the DF/AF. Use 'strict' to get an even stricter filtering
            'grad_thresh': 3,
            'grad_nfa': False,  # If True, use the image gradient and the NFA score of LSD to further threshold lines. We recommand using it for easy images, but to turn it off for challenging images (e.g. night, foggy, blurry images)
        }
    }

    # Load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = '/home/neeloy/projects/cs445/final_project/DeepLSD/weights/deeplsd_md.tar'
    ckpt = torch.load(str(ckpt), map_location='cpu')
    net = DeepLSD(conf)
    net.load_state_dict(ckpt['model'])
    net = net.to(device).eval()

    # Iterate over video paths
    for line_num, line in enumerate(tqdm(f)):

        # Check if raw frames folder exists for current video
        vid_path = os.path.join(line.strip())
        vid_file_name = (os.path.split(vid_path)[1]).split(".")[0]
        vid_raw_frames_dir = os.path.join(args.raw_frames, vid_file_name)
        if not os.path.exists(vid_raw_frames_dir):
            raise Exception

        # Variables for saving results
        curr_save_path = os.path.join(args.output, "{}.pkl".format(vid_file_name))
        save_dict = {}

        # Load video frames and predict fields
        frame_id = 0

        while(True): 

            curr_frame_path = os.path.join(vid_raw_frames_dir, "{}.png".format(frame_id))
            if not os.path.isfile(curr_frame_path):
                break
            curr_frame = cv2.imread(curr_frame_path)[:, :, ::-1]    # (H, W, 3)
            gray_img = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY) # (H, W)

            net_inputs = {'image': torch.tensor(gray_img, dtype=torch.float, device=device)[None, None] / 255.}
            with torch.no_grad():
                out = net(net_inputs)
                pred_lines = out['lines'][0]                    # (num_pred_lines, 2, 2)
                pred_lines_flat = pred_lines.reshape((-1, 4))   # (num_pred_lines, 4), classifier expects flattened input

            save_dict[frame_id] = pred_lines_flat

            frame_id += 1

        # Save predictions
        with open(curr_save_path, 'wb') as handle:
            pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

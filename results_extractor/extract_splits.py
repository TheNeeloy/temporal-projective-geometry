# Standard Imports
import os
import pickle
import argparse
import json

# Third Party Imports
import numpy as np


def get_parser():
    parser = argparse.ArgumentParser(description="Extract Data Splits by Ease of Classification")
    parser.add_argument(
        "--baseline_results", 
        default="/home/neeloy/projects/cs445/final_project/temporal-projective-geometry/baseline/results/all", 
        help="Path to folder with baseline results to extract splits for"
    )
    parser.add_argument(
        "--output", 
        default="/home/neeloy/projects/cs445/final_project/temporal-projective-geometry/data", 
        help="Path to folder to save splits in"
    )
    return parser

if __name__ == "__main__":

    args = get_parser().parse_args()

    # Check if all given paths exist
    assert os.path.exists(args.baseline_results)

    # Create save folder
    save_dir = os.path.join(args.output, "splits")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    sub_save_name = str(os.path.join(args.baseline_results)).split('/')[-1]
    sub_save_file = os.path.join(save_dir, "{}.json".format(sub_save_name))

    # Load results
    per_vid_pkl_path = os.path.join(args.baseline_results, "per_vid_results.pkl")
    with open(per_vid_pkl_path, 'rb') as fp:
        per_vid_dict = pickle.load(fp)

    # Dict to save splits
    save_dict = {}

    # Iterate over real and fake types
    for label in per_vid_dict:

        curr_label = 0 if label=="real" else 1
        save_dict[label] = {"easy":[], "medium":[], "hard":[]}

        # Compute baseline accuracy per video
        for video_name in per_vid_dict[label]:

            predictions = np.array(per_vid_dict[label][video_name]["pred"])
            labels = np.array([curr_label for _ in predictions])
            accuracy = (labels == predictions).sum() / labels.shape[0]

            if accuracy > 0.9:
                save_dict[label]["easy"].append(video_name)
            elif accuracy > 0.5:
                save_dict[label]["medium"].append(video_name)
            else:
                save_dict[label]["hard"].append(video_name)

    with open(sub_save_file, "w") as fp: 
        json.dump(save_dict, fp, indent=4)

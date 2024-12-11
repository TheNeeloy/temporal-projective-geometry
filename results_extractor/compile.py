# Standard Imports
import os
import pickle
import argparse
import json

# Third Party Imports
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def get_parser():
    parser = argparse.ArgumentParser(description="Compile Results Together")
    parser.add_argument(
        "--splits", 
        default="/home/neeloy/projects/cs445/final_project/temporal-projective-geometry/data/splits", 
        help="Path to folder with splits"
    )
    parser.add_argument(
        "--obj_shadow_results", 
        default="/home/neeloy/projects/cs445/final_project/temporal-projective-geometry/object_shadow/projective-geometry/results", 
        help="Path to folder with object-shadow anomaly classification model results"
    )
    parser.add_argument(
        "--perspective_results", 
        default="/home/neeloy/projects/cs445/final_project/temporal-projective-geometry/perspective_fields/projective_geometry/results", 
        help="Path to folder with perspective fields anomaly classification model results"
    )
    parser.add_argument(
        "--line_results", 
        default="/home/neeloy/projects/cs445/final_project/temporal-projective-geometry/line_segment/projective_geometry/results", 
        help="Path to folder with line segment anomaly classification model results"
    )
    parser.add_argument(
        "--baseline_results", 
        default="/home/neeloy/projects/cs445/final_project/temporal-projective-geometry/baseline/results", 
        help="Path to folder with baseline anomaly classification model results"
    )
    parser.add_argument(
        '--only_key_frames', 
        action=argparse.BooleanOptionalAction,
        help="Only compile results for key frames"
    )
    return parser

if __name__ == "__main__":

    args = get_parser().parse_args()

    # Check if all given paths exist
    high_split = "key_frames" if args.only_key_frames else "all"
    baseline_pkl_path = os.path.join(args.baseline_results, high_split, "per_vid_results.pkl")
    assert os.path.isfile(baseline_pkl_path)
    obj_shadow_pkl_path = os.path.join(args.obj_shadow_results, high_split, "per_vid_results.pkl")
    assert os.path.isfile(obj_shadow_pkl_path)
    perspective_pkl_path = os.path.join(args.perspective_results, high_split, "per_vid_results.pkl")
    assert os.path.isfile(perspective_pkl_path)
    line_pkl_path = os.path.join(args.line_results, high_split, "per_vid_results.pkl")
    assert os.path.isfile(line_pkl_path)
    splits_json_path = os.path.join(args.splits, "{}.json".format("key_frames" if args.only_key_frames else "all"))
    assert os.path.isfile(splits_json_path)

    # Load results
    with open(baseline_pkl_path, 'rb') as fp:
        baseline_dict = pickle.load(fp)
    with open(obj_shadow_pkl_path, 'rb') as fp:
        obj_shadow_dict = pickle.load(fp)
    with open(perspective_pkl_path, 'rb') as fp:
        perspective_dict = pickle.load(fp)
    with open(line_pkl_path, 'rb') as fp:
        line_dict = pickle.load(fp)
    with open(splits_json_path, 'r') as fp:
        splits_dict = json.load(fp)

    # Create save folder
    sub_save_dir = os.path.join("results", "key_frames" if args.only_key_frames else "all")
    if not os.path.exists(sub_save_dir):
        os.makedirs(sub_save_dir)

    # Dict to store labels and predictions per model per split
    results_dict = {}
    split_names = ["complete", "easy", "medium", "hard"]
    model_names = ["baseline", "obj_shadow", "perspective", "lines"]
    for split_name in split_names:
        results_dict[split_name] = {}
        for model_name in model_names:
            results_dict[split_name][model_name] = {}
            results_dict[split_name][model_name]["labels"] = []
            results_dict[split_name][model_name]["prob"] = []

    # Iterate over real and fake types
    for label in splits_dict:

        curr_label = 0 if label=="real" else 1

        # Iterate over easy, medium, and hard splits
        for split_difficulty in splits_dict[label]:

            # Iterate over videos
            for video_name in splits_dict[label][split_difficulty]:

                baseline_predictions = baseline_dict[label][video_name]["probs"]
                baseline_labels = [curr_label for _ in baseline_predictions]
                results_dict[split_difficulty]["baseline"]["prob"].extend(baseline_predictions)
                results_dict[split_difficulty]["baseline"]["labels"].extend(baseline_labels)
                results_dict["complete"]["baseline"]["prob"].extend(baseline_predictions)
                results_dict["complete"]["baseline"]["labels"].extend(baseline_labels)

                obj_shadow_predictions = obj_shadow_dict[label][video_name]["probs"]
                obj_shadow_labels = [curr_label for _ in obj_shadow_predictions]
                results_dict[split_difficulty]["obj_shadow"]["prob"].extend(obj_shadow_predictions)
                results_dict[split_difficulty]["obj_shadow"]["labels"].extend(obj_shadow_labels)
                results_dict["complete"]["obj_shadow"]["prob"].extend(obj_shadow_predictions)
                results_dict["complete"]["obj_shadow"]["labels"].extend(obj_shadow_labels)

                perspective_predictions = perspective_dict[label][video_name]["probs"]
                perspective_labels = [curr_label for _ in perspective_predictions]
                results_dict[split_difficulty]["perspective"]["prob"].extend(perspective_predictions)
                results_dict[split_difficulty]["perspective"]["labels"].extend(perspective_labels)
                results_dict["complete"]["perspective"]["prob"].extend(perspective_predictions)
                results_dict["complete"]["perspective"]["labels"].extend(perspective_labels)

                line_predictions = line_dict[label][video_name]["probs"]
                line_labels = [curr_label for _ in line_predictions]
                results_dict[split_difficulty]["lines"]["prob"].extend(line_predictions)
                results_dict[split_difficulty]["lines"]["labels"].extend(line_labels)
                results_dict["complete"]["lines"]["prob"].extend(line_predictions)
                results_dict["complete"]["lines"]["labels"].extend(line_labels)

    # Save compiled results
    with open(os.path.join(sub_save_dir, "compiled.json"), "w") as fp: 
        json.dump(results_dict, fp, indent=4)

    # Create ROC plots
    for split_name in results_dict:

        plt.figure()
        lw = 2

        for model_name in results_dict[split_name]:

            prob = results_dict[split_name][model_name]["prob"]
            labels = results_dict[split_name][model_name]["labels"]

            # Compute ROC curve and ROC area for each class
            fpr, tpr, _ = roc_curve(labels, prob)
            roc_auc = auc(fpr, tpr)

            # Plotting the ROC curve
            area_formatted = "{0:.2f}".format(roc_auc)
            plt.plot(fpr, tpr, lw=lw, label="{} (area = {})".format(model_name, area_formatted))

        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title("ROC on {} Split".format(split_name))
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(sub_save_dir, "{}_split_roc.png".format(split_name)), dpi=300)

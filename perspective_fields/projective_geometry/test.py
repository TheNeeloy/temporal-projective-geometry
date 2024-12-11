# Standard Imports
import pickle
import os
import argparse
import json

# Third Party Imports
import torch, torchvision
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import torchvision.transforms

# Local Imports
from fields_model import load_model


def transform_maps(latitude_map, gravity_maps):
    latitude_map = latitude_map / 90.0
    joined_maps = torch.cat([latitude_map.unsqueeze(0), gravity_maps], dim = 0)
    return joined_maps

def get_parser():
    parser = argparse.ArgumentParser(description="Perspective Fields Hallucination Classifier")
    parser.add_argument(
        "--fake_input", 
        default="/home/neeloy/projects/cs445/final_project/temporal-projective-geometry/data/fake_video_paths.txt", 
        help="Path to txt file with paths to fake videos in dataset"
    )
    parser.add_argument(
        "--real_input", 
        default="/home/neeloy/projects/cs445/final_project/temporal-projective-geometry/data/real_video_paths.txt", 
        help="Path to txt file with paths to real videos in dataset"
    )
    parser.add_argument(
        "--key_frames", 
        default="/home/neeloy/projects/cs445/final_project/temporal-projective-geometry/data/key_frames", 
        help="Path to folder with jsons of key frames per video"
    )
    parser.add_argument(
        "--fields", 
        default="/home/neeloy/projects/cs445/final_project/temporal-projective-geometry/data/perspective_fields", 
        help="Path to folder with perspective fields per video frame"
    )
    parser.add_argument(
        '--only_key_frames', 
        action=argparse.BooleanOptionalAction,
        help="Only perform inference on key frames"
    )
    return parser

if __name__ == "__main__":

    args = get_parser().parse_args()

    # Check if all given paths exist
    assert os.path.isfile(args.fake_input)
    assert os.path.isfile(args.real_input)
    assert os.path.exists(args.key_frames)
    assert os.path.exists(args.fields)

    # Create save folder
    sub_save_dir = os.path.join("results", "key_frames" if args.only_key_frames else "all")
    if not os.path.exists(sub_save_dir):
        os.makedirs(sub_save_dir)

    # Transform for image inputs and set device
    toTensor = torchvision.transforms.ToTensor()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load classification model
    ckpt_path = os.path.join("checkpoints/Fields_combined.pt")
    model = load_model(target_device=device, path_to_checkpoint=ckpt_path)
    model.eval()

    # Tensors to store results
    all_predicted = torch.tensor([]).to(device)
    all_labels = torch.tensor([]).to(device)
    all_pred_probs = torch.tensor([]).to(device)
    correct = 0
    total = 0
    predictions_dict = {"real":{}, "fake":{}}

    # First predict scores for real videos and then fake videos
    for label in range(2):  # 0 when real, 1 when fake

        if label:
            f = open(args.fake_input, "r")
        else:
            f = open(args.real_input, "r")
        curr_label_torch = torch.tensor([label], dtype=torch.int64, requires_grad=False).to(device)

        sub_fields_dir = os.path.join(args.fields, "fake" if label else "real")
        sub_key_frames_dir = os.path.join(args.key_frames, "fake" if label else "real")

        # Iterate over video paths
        for line in tqdm(f):

            # Check if necessary files exist for current video
            vid_path = os.path.join(line.strip())
            vid_file_name = (os.path.split(vid_path)[1]).split(".")[0]
            vid_fields_dir = os.path.join(sub_fields_dir, vid_file_name)
            vid_key_frames = os.path.join(sub_key_frames_dir, "{}.json".format(vid_file_name))
            assert os.path.isfile(vid_key_frames)
            with open(vid_key_frames, 'r') as fp:
                vid_key_frames_dict = json.load(fp)

            # Initialize predictions dict for video
            predictions_dict["fake" if label else "real"][vid_file_name] = {"frame_ids":[], "probs":[], "pred":[]}

            # Collect frame IDs to predict for
            if args.only_key_frames:
                valid_frame_ids = []
                for k in sorted(vid_key_frames_dict.keys()):
                    key_frame = vid_key_frames_dict[k]["key_frame"]
                    valid_frame_ids.append(key_frame)
                    for detail_frame in vid_key_frames_dict[k]["detail_frames"]:
                        valid_frame_ids.append(detail_frame)
            else:
                num_frames = len([name for name in os.listdir(vid_fields_dir) if os.path.isfile(os.path.join(vid_fields_dir, name))])
                valid_frame_ids = [i for i in range(num_frames)]

            # Predict anomaly scores per frame
            for frame_id in valid_frame_ids:

                curr_field_filepath = os.path.join(vid_fields_dir, "{}.pt".format(frame_id))

                curr_field = torch.load(curr_field_filepath)
                
                latitude_map = curr_field['pred_latitude_original']
                gravity_maps = curr_field['pred_gravity_original']
                joined_maps = transform_maps(latitude_map,  gravity_maps)
                joined_maps = joined_maps.unsqueeze(0).to(device)   # (1, 3, H, W)
        
                with torch.no_grad():

                    outputs = model(joined_maps)    # (1, 2), scores for normal and abnormal
                    _, predicted = torch.max(outputs.data, 1)
                    total += 1
                    correct += (predicted == curr_label_torch).sum().item()
                    all_predicted = torch.cat((all_predicted, predicted))
                    all_labels = torch.cat((all_labels, curr_label_torch))
                    all_pred_probs = torch.cat((all_pred_probs, outputs.data))
                    curr_anom_pred_prob = outputs[:,1].cpu().item()
                    curr_pred = predicted.cpu().item()

                    predictions_dict["fake" if label else "real"][vid_file_name]["frame_ids"].append(frame_id)
                    predictions_dict["fake" if label else "real"][vid_file_name]["probs"].append(curr_anom_pred_prob)
                    predictions_dict["fake" if label else "real"][vid_file_name]["pred"].append(curr_pred)

        # Close txt file
        f.close()

    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(all_labels.cpu(), all_pred_probs[:,1].cpu())
    roc_auc = auc(fpr, tpr)

    # Plotting the ROC curve
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plot_name = "Key Frames" if args.only_key_frames else "All Frames"
    plt.title(f'ROC: {plot_name}')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(sub_save_dir, "roc.png"), dpi=300)

    # Confusion matrix
    conf_matrix = confusion_matrix(all_labels.cpu(), all_predicted.cpu())
    print(f"{plot_name}")
    print("ROC curve area:", roc_auc)
    print(conf_matrix)
    print(f"{conf_matrix[0].sum().item()} generated images, {conf_matrix[1].sum().item()} real images")
    tn = conf_matrix[0,0]
    tp = conf_matrix[1,1]
    fp = conf_matrix[0,1]
    fn = conf_matrix[1,0]
    print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
    print(f"Precision: {tp/(tp+fp)}, Recall: {tp/(tp+fn)}")
    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy}")
    print()

    # Saving overall results
    results_path = os.path.join(sub_save_dir, "results.pkl")
    result_dict = {'all_predicted':all_predicted.cpu(), 'all_labels':all_labels.cpu(), 
                'all_pred_probs':all_pred_probs.cpu(), 'correct':correct, 'total':total,
                'fpr':fpr, 'tpr':tpr, 'roc_auc':roc_auc}
    with open(results_path, 'wb') as handle:
        pickle.dump(result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Saving per-video results
    per_vid_results_path = os.path.join(sub_save_dir, "per_vid_results.pkl")
    with open(per_vid_results_path, 'wb') as handle:
        pickle.dump(predictions_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

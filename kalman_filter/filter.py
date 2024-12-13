# Standard Imports
import os
import pickle
import argparse
import json

# Third Party Imports
import numpy as np
from tqdm import tqdm
from scipy.stats import multivariate_normal
from scipy.signal import butter, lfilter


def get_parser():
    parser = argparse.ArgumentParser(description="Kalman Filter Scoring")
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
        "--raw_frames", 
        default="/home/neeloy/projects/cs445/final_project/temporal-projective-geometry/data/raw_frames", 
        help="Path to folder with raw frames for videos"
    )
    parser.add_argument(
        "--key_frames", 
        default="/home/neeloy/projects/cs445/final_project/temporal-projective-geometry/data/key_frames", 
        help="Path to folder with jsons of key frames per video"
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
        '--only_key_frames', 
        action=argparse.BooleanOptionalAction,
        help="Only compile results for key frames"
    )
    parser.add_argument(
        '--smooth', 
        action=argparse.BooleanOptionalAction,
        help="Smooth each models' predictions before applying kalman filter"
    )
    return parser

if __name__ == "__main__":

    args = get_parser().parse_args()

    # Check if all given paths exist
    high_split = "key_frames" if args.only_key_frames else "all"
    assert os.path.exists(args.raw_frames)
    assert os.path.exists(args.key_frames)
    obj_shadow_pkl_path = os.path.join(args.obj_shadow_results, high_split, "per_vid_results.pkl")
    assert os.path.isfile(obj_shadow_pkl_path)
    perspective_pkl_path = os.path.join(args.perspective_results, high_split, "per_vid_results.pkl")
    assert os.path.isfile(perspective_pkl_path)
    line_pkl_path = os.path.join(args.line_results, high_split, "per_vid_results.pkl")
    assert os.path.isfile(line_pkl_path)

    # Load results
    with open(obj_shadow_pkl_path, 'rb') as fp:
        obj_shadow_dict = pickle.load(fp)
    with open(perspective_pkl_path, 'rb') as fp:
        perspective_dict = pickle.load(fp)
    with open(line_pkl_path, 'rb') as fp:
        line_dict = pickle.load(fp)

    # Filter to smooth predictions
    b, a = butter(2, 0.2, fs=10)

    # Calculate normalization factors for raw probs output by models 
    model_names = ["obj_shadow", "perspective", "lines"]
    norm_dict = {}
    for model_name in model_names:
        norm_dict[model_name] = {"min":float('inf'), "max":-float('inf')}
    for label in obj_shadow_dict:
        for video_name in obj_shadow_dict[label]:
            probs = obj_shadow_dict[label][video_name]["probs"]
            if args.smooth:
                probs_smooth = lfilter(b, a, probs)
                obj_shadow_dict[label][video_name]["probs"] = probs_smooth
            if probs:
                norm_dict["obj_shadow"]["min"] = min(norm_dict["obj_shadow"]["min"], np.min(probs))
                norm_dict["obj_shadow"]["max"] = max(norm_dict["obj_shadow"]["max"], np.max(probs))
    for label in perspective_dict:
        for video_name in perspective_dict[label]:
            probs = perspective_dict[label][video_name]["probs"]
            if args.smooth:
                probs_smooth = lfilter(b, a, probs)
                perspective_dict[label][video_name]["probs"] = probs_smooth
            if probs:
                norm_dict["perspective"]["min"] = min(norm_dict["perspective"]["min"], np.min(probs))
                norm_dict["perspective"]["max"] = max(norm_dict["perspective"]["max"], np.max(probs))
    for label in line_dict:
        for video_name in line_dict[label]:
            probs = line_dict[label][video_name]["probs"]
            if args.smooth:
                probs_smooth = lfilter(b, a, probs)
                line_dict[label][video_name]["probs"] = probs_smooth
            if probs:
                norm_dict["lines"]["min"] = min(norm_dict["lines"]["min"], np.min(probs))
                norm_dict["lines"]["max"] = max(norm_dict["lines"]["max"], np.max(probs))

    # Create save folder
    sub_save_dir = os.path.join("results", "key_frames" if args.only_key_frames else "all")
    if not os.path.exists(sub_save_dir):
        os.makedirs(sub_save_dir)

    # Dict to store results
    predictions_dict = {"real":{}, "fake":{}}

    # Predict scores for real and fake videos
    for label in predictions_dict:

        curr_label = 0 if label=="real" else 1
        if curr_label:
            f = open(args.fake_input, "r")
        else:
            f = open(args.real_input, "r")

        sub_key_frames_dir = os.path.join(args.key_frames, label)
        sub_raw_frames_dir = os.path.join(args.raw_frames, label)

        # Iterate over video paths
        for line in tqdm(f):

            # Get current video name and key frames
            vid_path = os.path.join(line.strip())
            vid_file_name = (os.path.split(vid_path)[1]).split(".")[0]
            vid_key_frames = os.path.join(sub_key_frames_dir, "{}.json".format(vid_file_name))
            assert os.path.isfile(vid_key_frames)
            with open(vid_key_frames, 'r') as fp:
                vid_key_frames_dict = json.load(fp)

            # Initialize predictions dict for video
            predictions_dict[label][vid_file_name] = {"frame_ids":[], "probs":[], "norm_probs":[], "pred":[]}

            # Collect frame IDs to predict for
            vid_frames_path = os.path.join(sub_raw_frames_dir, vid_file_name)
            num_frames = len([name for name in os.listdir(vid_frames_path) if os.path.isfile(os.path.join(vid_frames_path, name))])
            if args.only_key_frames:
                valid_frame_ids = []
                for k in sorted(vid_key_frames_dict.keys()):
                    key_frame = vid_key_frames_dict[k]["key_frame"]
                    valid_frame_ids.append(key_frame)
                    for detail_frame in vid_key_frames_dict[k]["detail_frames"]:
                        valid_frame_ids.append(detail_frame)
            else:
                valid_frame_ids = [i for i in range(num_frames)]

            # Get predictions per module for video
            obj_shadow_preds = obj_shadow_dict[label][vid_file_name]["pred"]
            perspective_preds = perspective_dict[label][vid_file_name]["pred"]
            line_preds = line_dict[label][vid_file_name]["pred"]
            obj_shadow_probs = obj_shadow_dict[label][vid_file_name]["probs"]
            perspective_probs = perspective_dict[label][vid_file_name]["probs"]
            line_probs = line_dict[label][vid_file_name]["probs"]
            obj_shadow_frame_ids = obj_shadow_dict[label][vid_file_name]["frame_ids"]
            perspective_frame_ids = perspective_dict[label][vid_file_name]["frame_ids"]
            line_frame_ids = line_dict[label][vid_file_name]["frame_ids"]

            ## Kalman filter initialization ##
            
            # Prediction step
            A = np.array([[1.   , 0.   , 0.   , 0.],            # process dynamics matrix
                          [0.   , 1.   , 0.   , 0.],
                          [0.   , 0.   , 1.   , 0.],
                          [1./3., 1./3., 1./3., 0.]])
            q_mu = np.zeros((4))                                # process noise mean
            q_cov = np.eye(4) * 0.1                             # process noise covariance
            w_distr = multivariate_normal(cov=q_cov, mean=q_mu) # process noise distribution

            # Measurement update step
            r_mu = 0.   # observation noise mean
            r_cov = 1.  # observation noise variance

            # Estimate initializations
            x_pred, x_prob, x_prob_norm = np.zeros((4,1)), np.zeros((4,1)), np.zeros((4,1))             # state estimate
            p_cov_pred, p_cov_prob, p_cov_prob_norm = np.eye(4) * 0.1, np.eye(4) * 0.1, np.eye(4) * 0.1 # a posteriori estimate (changes with every prediction and measurement update)
            
            # Observation matrices
            h_obj_shadow = np.array([[1., 0., 0., 0.]])
            h_perspective = np.array([[0., 1., 0., 0.]])
            h_lines = np.array([[0., 0., 1., 0.]])

            # Variable to track if all sensors have been seen
            seen = False

            ##################################

            # Iterate over timesteps
            for frame_id in range(num_frames):

                # Initial timestep
                if not seen:

                    # Which sensors are available for this timestep
                    seen_obj_shadow = frame_id in obj_shadow_frame_ids
                    seen_perspective = frame_id in perspective_frame_ids
                    seen_lines = frame_id in line_frame_ids
                    valid_id = frame_id in valid_frame_ids

                    # Still not valid timestep yet
                    if not valid_id or not (seen_obj_shadow and seen_perspective and seen_lines):
                        continue

                    # Initialize kalman filter
                    else:

                        # Update seen sensors flag
                        seen = True

                        ## Measure sensors

                        # obj-shadow model
                        obj_shadow_index = obj_shadow_frame_ids.index(frame_id)
                        obj_shadow_meas_pred = obj_shadow_preds[obj_shadow_index]
                        obj_shadow_meas_prob = obj_shadow_probs[obj_shadow_index]
                        obj_shadow_meas_prob_norm = (obj_shadow_meas_prob - norm_dict["obj_shadow"]["min"]) / (norm_dict["obj_shadow"]["max"] - norm_dict["obj_shadow"]["min"])

                        # perspective model
                        perspective_index = perspective_frame_ids.index(frame_id)
                        perspective_meas_pred = perspective_preds[perspective_index]
                        perspective_meas_prob = perspective_probs[perspective_index]
                        perspective_meas_prob_norm = (perspective_meas_prob - norm_dict["perspective"]["min"]) / (norm_dict["perspective"]["max"] - norm_dict["perspective"]["min"])

                        # line segment model
                        line_index = line_frame_ids.index(frame_id)
                        line_meas_pred = line_preds[line_index]
                        line_meas_prob = line_probs[line_index]
                        line_meas_prob_norm = (line_meas_prob - norm_dict["lines"]["min"]) / (norm_dict["lines"]["max"] - norm_dict["lines"]["min"])

                        # Initial anomaly estimate
                        init_anom_pred = np.mean([obj_shadow_meas_pred, perspective_meas_pred, line_meas_pred])
                        init_anom_prob = np.mean([obj_shadow_meas_prob, perspective_meas_prob, line_meas_prob])
                        init_anom_prob_norm = np.mean([obj_shadow_meas_prob_norm, perspective_meas_prob_norm, line_meas_prob_norm])

                        # Complete initial states
                        x_pred = np.array([[obj_shadow_meas_pred ],
                                           [perspective_meas_pred],
                                           [line_meas_pred       ],
                                           [init_anom_pred       ]])
                        x_prob = np.array([[obj_shadow_meas_prob ],
                                           [perspective_meas_prob],
                                           [line_meas_prob       ],
                                           [init_anom_prob       ]])
                        x_prob_norm = np.array([[obj_shadow_meas_prob_norm ],
                                                [perspective_meas_prob_norm],
                                                [line_meas_prob_norm       ],
                                                [init_anom_prob_norm       ]])

                        # Add predictions
                        predictions_dict[label][vid_file_name]["frame_ids"].append(frame_id)
                        predictions_dict[label][vid_file_name]["probs"].append(init_anom_prob)
                        predictions_dict[label][vid_file_name]["norm_probs"].append(init_anom_prob_norm)
                        predictions_dict[label][vid_file_name]["pred"].append(init_anom_pred)

                # Every future time
                else:

                    ## Prediction step

                    # Sample process noise
                    w_samples = w_distr.rvs(size=3)

                    # Propagate state with noise
                    x_pred = A @ x_pred + w_samples[0][np.newaxis].T
                    x_prob = A @ x_prob + w_samples[1][np.newaxis].T
                    x_prob_norm = A @ x_prob_norm + w_samples[2][np.newaxis].T

                    # Update error estimate
                    p_cov_pred = A @ p_cov_pred @ A.T + q_cov
                    p_cov_prob = A @ p_cov_prob @ A.T + q_cov
                    p_cov_prob_norm = A @ p_cov_prob_norm @ A.T + q_cov

                    ## Update step

                    # Which sensors are available for this timestep
                    seen_obj_shadow = frame_id in obj_shadow_frame_ids
                    seen_perspective = frame_id in perspective_frame_ids
                    seen_lines = frame_id in line_frame_ids
                    valid_id = frame_id in valid_frame_ids

                    # Measurement update not possible without any sensor measurements for this timestep
                    if not valid_id or not (seen_obj_shadow or seen_perspective or seen_lines):
                        continue

                    # obj-shadow model
                    if seen_obj_shadow:

                        # Get sensor measurements
                        obj_shadow_index = obj_shadow_frame_ids.index(frame_id)
                        obj_shadow_meas_pred = obj_shadow_preds[obj_shadow_index]
                        obj_shadow_meas_prob = obj_shadow_probs[obj_shadow_index]
                        obj_shadow_meas_prob_norm = (obj_shadow_meas_prob - norm_dict["obj_shadow"]["min"]) / (norm_dict["obj_shadow"]["max"] - norm_dict["obj_shadow"]["min"])

                        # Compute kalman gain
                        k_gain_pred = (p_cov_pred @ h_obj_shadow.T) * ((1.)/(h_obj_shadow @ p_cov_pred @ h_obj_shadow.T + r_cov))
                        k_gain_prob = (p_cov_prob @ h_obj_shadow.T) * ((1.)/(h_obj_shadow @ p_cov_prob @ h_obj_shadow.T + r_cov))
                        k_gain_prob_norm = (p_cov_prob_norm @ h_obj_shadow.T) * ((1.)/(h_obj_shadow @ p_cov_prob_norm @ h_obj_shadow.T + r_cov))

                        # Update estimate of state
                        x_pred = x_pred + k_gain_pred * (obj_shadow_meas_pred - (h_obj_shadow @ x_pred)[0,0] + np.random.normal(r_mu, r_cov))
                        x_prob = x_prob + k_gain_prob * (obj_shadow_meas_prob - (h_obj_shadow @ x_prob)[0,0] + np.random.normal(r_mu, r_cov))
                        x_prob_norm = x_prob_norm + k_gain_prob_norm * (obj_shadow_meas_prob_norm - (h_obj_shadow @ x_prob_norm)[0,0] + np.random.normal(r_mu, r_cov))

                        # Update estimate of error
                        p_cov_pred = (np.eye(4) - k_gain_pred @ h_obj_shadow) @ p_cov_pred
                        p_cov_prob = (np.eye(4) - k_gain_prob @ h_obj_shadow) @ p_cov_prob
                        p_cov_prob_norm = (np.eye(4) - k_gain_prob_norm @ h_obj_shadow) @ p_cov_prob_norm

                    # perspective model
                    if seen_perspective:

                        # Get sensor measurements
                        perspective_index = perspective_frame_ids.index(frame_id)
                        perspective_meas_pred = perspective_preds[perspective_index]
                        perspective_meas_prob = perspective_probs[perspective_index]
                        perspective_meas_prob_norm = (perspective_meas_prob - norm_dict["perspective"]["min"]) / (norm_dict["perspective"]["max"] - norm_dict["perspective"]["min"])

                        # Compute kalman gain
                        k_gain_pred = (p_cov_pred @ h_perspective.T) * ((1.)/(h_perspective @ p_cov_pred @ h_perspective.T + r_cov))
                        k_gain_prob = (p_cov_prob @ h_perspective.T) * ((1.)/(h_perspective @ p_cov_prob @ h_perspective.T + r_cov))
                        k_gain_prob_norm = (p_cov_prob_norm @ h_perspective.T) * ((1.)/(h_perspective @ p_cov_prob_norm @ h_perspective.T + r_cov))

                        # Update estimate of state
                        x_pred = x_pred + k_gain_pred * (perspective_meas_pred - (h_perspective @ x_pred)[0,0] + np.random.normal(r_mu, r_cov))
                        x_prob = x_prob + k_gain_prob * (perspective_meas_prob - (h_perspective @ x_prob)[0,0] + np.random.normal(r_mu, r_cov))
                        x_prob_norm = x_prob_norm + k_gain_prob_norm * (perspective_meas_prob_norm - (h_perspective @ x_prob_norm)[0,0] + np.random.normal(r_mu, r_cov))

                        # Update estimate of error
                        p_cov_pred = (np.eye(4) - k_gain_pred @ h_perspective) @ p_cov_pred
                        p_cov_prob = (np.eye(4) - k_gain_prob @ h_perspective) @ p_cov_prob
                        p_cov_prob_norm = (np.eye(4) - k_gain_prob_norm @ h_perspective) @ p_cov_prob_norm

                    # line segment model
                    if seen_lines:

                        # Get sensor measurements
                        line_index = line_frame_ids.index(frame_id)
                        line_meas_pred = line_preds[line_index]
                        line_meas_prob = line_probs[line_index]
                        line_meas_prob_norm = (line_meas_prob - norm_dict["lines"]["min"]) / (norm_dict["lines"]["max"] - norm_dict["lines"]["min"])

                        # Compute kalman gain
                        k_gain_pred = (p_cov_pred @ h_lines.T) * ((1.)/(h_lines @ p_cov_pred @ h_lines.T + r_cov))
                        k_gain_prob = (p_cov_prob @ h_lines.T) * ((1.)/(h_lines @ p_cov_prob @ h_lines.T + r_cov))
                        k_gain_prob_norm = (p_cov_prob_norm @ h_lines.T) * ((1.)/(h_lines @ p_cov_prob_norm @ h_lines.T + r_cov))

                        # Update estimate of state
                        x_pred = x_pred + k_gain_pred * (line_meas_pred - (h_lines @ x_pred)[0,0] + np.random.normal(r_mu, r_cov))
                        x_prob = x_prob + k_gain_prob * (line_meas_prob - (h_lines @ x_prob)[0,0] + np.random.normal(r_mu, r_cov))
                        x_prob_norm = x_prob_norm + k_gain_prob_norm * (line_meas_prob_norm - (h_lines @ x_prob_norm)[0,0] + np.random.normal(r_mu, r_cov))

                        # Update estimate of error
                        p_cov_pred = (np.eye(4) - k_gain_pred @ h_lines) @ p_cov_pred
                        p_cov_prob = (np.eye(4) - k_gain_prob @ h_lines) @ p_cov_prob
                        p_cov_prob_norm = (np.eye(4) - k_gain_prob_norm @ h_lines) @ p_cov_prob_norm

                    # Add predictions
                    predictions_dict[label][vid_file_name]["frame_ids"].append(frame_id)
                    predictions_dict[label][vid_file_name]["probs"].append(x_prob[3,0])
                    predictions_dict[label][vid_file_name]["norm_probs"].append(x_prob_norm[3,0])
                    predictions_dict[label][vid_file_name]["pred"].append(x_pred[3,0])

        # Close txt file
        f.close()

    # Saving per-video results
    per_vid_results_path = os.path.join(sub_save_dir, "per_vid_results.pkl")
    with open(per_vid_results_path, 'wb') as handle:
        pickle.dump(predictions_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

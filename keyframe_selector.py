# Standard Imports
import os
import json

# Third Party Imports
import cv2
import numpy as np
from tqdm import tqdm


# Constants
SKIP = 5        # Number of frames to initially skip when extracting from video
CUTOFF = 0.5    # Cutoff distance threshold when computing local densities
TOP_K = 10      # Number of keyframes to take from video before detailed frame extraction

if __name__ == "__main__":

    # Make save directory
    save_dir = os.path.join("data/key_frames")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    raw_frames_dir = os.path.join("data/raw_frames")

    # Path to txt files with videos
    video_paths_txt_list = [os.path.join("data/fake_video_paths.txt"), os.path.join("data/real_video_paths.txt")]

    # Iterate txt files
    for txt_id, video_paths_txt in enumerate(video_paths_txt_list):
        
        # Check if folder for video type exists
        sub_raw_frames_dir = os.path.join(raw_frames_dir, "fake" if txt_id==0 else "real")
        if not os.path.exists(sub_raw_frames_dir):
            raise Exception

        # Make save sub-directory
        sub_save_dir = os.path.join(save_dir, "fake" if txt_id==0 else "real")
        if not os.path.exists(sub_save_dir):
            os.makedirs(sub_save_dir)

        f = open(video_paths_txt, "r")

        # Iterate over video paths
        for line in tqdm(f):

            # Check if raw frames folder exists for current video
            vid_path = os.path.join(line.strip())
            file_name = (os.path.split(vid_path)[1]).split(".")[0]
            vid_raw_frames_dir = os.path.join(sub_raw_frames_dir, file_name)
            if not os.path.exists(vid_raw_frames_dir):
                raise Exception

            # Load video frames skipping every few
            frame_id = 0
            frames = []
            frame_ids = []
            while(True): 
                curr_frame_path = os.path.join(vid_raw_frames_dir, "{}.png".format(frame_id))
                if not os.path.isfile(curr_frame_path):
                    break
                curr_frame = cv2.imread(curr_frame_path, cv2.IMREAD_GRAYSCALE)  # (H, W)
                frames.append(np.expand_dims(curr_frame, 0))
                frame_ids.append(frame_id)
                frame_id += SKIP
            frame_ids_np = np.array(frame_ids)

            # Compute distance (normalized cross correlation) between each pair of images
            frames_np = np.concatenate(frames, axis=0).reshape((len(frames), -1))   # (num_frames, H*W)
            distances = 2. - (np.corrcoef(frames_np) + 1.)                          # (num_frames, num_frames) ; 
                                                                                    # 2 - (x + 1) gives range in [0, 2] with smaller values meaning closer features

            # Compute local density per image using gaussian kernel
            curr_mask = (1 - np.eye(len(frames)).reshape((-1))).astype(bool)                    # (num_frames*num_frames)
            distances_flat = distances.reshape((-1))                                            # (num_frames*num_frames)
            distances_masked = distances_flat[curr_mask].reshape((len(frames), len(frames)-1))  # (num_frames, num_frames-1) ; ignores NCC of each image with itself
            local_density = np.sum(np.exp(-np.square(distances_masked/CUTOFF)), axis=1)         # (num_frames)

            # Compute relative distances
            local_density_unsqueezed = local_density.reshape((-1, 1))               # (num_frames, 1)
            local_density_differences = local_density - local_density_unsqueezed    # (num_frames, num_frames)
            greater_local_density_differences_mask = local_density_differences > 0. # (num_frames, num_frames)

            # Compute delta for each frame based on the greatest relative differences
            max_rho_index = local_density.argmax()
            deltas = np.zeros((len(frames)))
            for frame_ind in range(len(frames)):
                if frame_ind == max_rho_index:
                    curr_distances = distances[frame_ind]
                    deltas[frame_ind] = curr_distances.max()
                else:
                    curr_distances = distances[frame_ind][greater_local_density_differences_mask[frame_ind]]
                    deltas[frame_ind] = curr_distances.min()

            # Compute final keyframe scores
            keyframe_scores = np.multiply(local_density, deltas)

            # Get top keyframes
            keyframe_inds = np.argpartition(keyframe_scores, -TOP_K)[-TOP_K:]
            keyframe_inds.sort()
            keyframe_ids = frame_ids_np[keyframe_inds]

            # Instantiate dict to hold keyframe clusters
            keyframe_cluster_dict = {}
            for i in range(TOP_K):
                keyframe_cluster_dict[i] = {}
                keyframe_cluster_dict[i]["key_frame"] = int(keyframe_ids[i])
                keyframe_cluster_dict[i]["detail_frames"] = []

            # Extract extra detail frames
            for k in range(TOP_K - 1):

                i, j = keyframe_inds[k], keyframe_inds[k+1]
                curr_dist = distances[i,j]

                # Should add detail frames between two keyframes since distance is large
                if curr_dist > CUTOFF:

                    t = i
                    for d in range(int(i+1), int(j)):
                        curr_dist = distances[t,d]
                        if curr_dist > CUTOFF:
                            t = d
                            keyframe_cluster_dict[k]["detail_frames"].append(int(frame_ids_np[d]))

            # Convert and write JSON object to file
            with open(os.path.join(sub_save_dir, "{}.json".format(file_name)), "w") as outfile: 
                json.dump(keyframe_cluster_dict, outfile, indent=4)

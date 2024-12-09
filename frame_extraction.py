# Standard Imports
import os

# Third Party Imports
import cv2


if __name__ == "__main__":

    # Make save directory
    save_dir = os.path.join("data/raw_frames")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Path to txt files with videos
    video_paths_txt_list = [os.path.join("data/fake_video_paths.txt"), os.path.join("data/real_video_paths.txt")]

    # Iterate txt files
    for i, video_paths_txt in enumerate(video_paths_txt_list):
        
        # Make save sub-directory
        sub_save_dir = os.path.join(save_dir, "fake" if i==0 else "real")
        if not os.path.exists(sub_save_dir):
            os.makedirs(sub_save_dir)

        f = open(video_paths_txt, "r")

        # Iterate over video paths
        for line in f:

            vid_path = os.path.join(line.strip())
            if not os.path.isfile(vid_path):
                raise Exception

            file_name = (os.path.split(vid_path)[1]).split(".")[0]
            
            # Make directory to hold frames of current video
            frames_save_dir = os.path.join(sub_save_dir, file_name)
            if not os.path.exists(frames_save_dir):
                os.makedirs(frames_save_dir)

            # Load video
            vid = cv2.VideoCapture(vid_path)

            # Extract frames 
            # https://www.geeksforgeeks.org/extract-images-from-video-in-python/
            frame_id = 0
            while(True): 
                
                ret, frame = vid.read() 
            
                # Frame is valid to save
                if ret: 
                    resized_frame = cv2.resize(frame, (256, 256), interpolation = cv2.INTER_CUBIC)
                    frame_path = os.path.join(frames_save_dir, "{}.png".format(frame_id))
                    cv2.imwrite(frame_path, resized_frame) # Save frame
                    frame_id += 1
                # Done reading video
                else: 
                    break
            
            # Release all space and windows once done 
            vid.release() 
            cv2.destroyAllWindows() 

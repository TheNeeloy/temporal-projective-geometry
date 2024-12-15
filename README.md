# Project Code for *An Exploration Into AI-Generated Video Hallucination Detection*

## Abstract

Recent generative models like Sora have shown promise in producing high quality images and videos given text input, but they can also hallucinate inconsistent, undesirable, and irrelevant features. 
Existing video hallucination detectors, like Sora Detector, leverage another language model for image captioning, which itself can hallucinate. 
Recent works have shown promising results in exploiting extracted geometries from static images to detect hallucinations, but have not been applied to video sequences.
In this work, we first present a short case study showcasing hallucinated predictions output by language models used for anomaly detection, motivating the need for structure-guided hallucination detectors.
We then utilize state-of-the-art models to extract geometric cues from a custom-curated dataset of AI-generated and real videos, and directly apply a recent geometry-based hallucination detector on each static frame.
Finally, we extend the static frame hallucination detector to consider the temporal consistency across consecutive frames by developing a Kalman filter that fuses predictions from different models. 

## File Structure

Look inside each sub-folder to see how to run that part of the project.
```
./
│   README.md - This file
│   frame_extraction.py - Saves raw frames from list of videos
|   keyframe_selector.py - Extracts key frames from videos
|   projective_geometry.yml - Conda environment file
│
└───baseline - Folder with code for pixel-based baseline classifier
│   
└───data - Folder where extracted data is saved
│   
└───kalman_filter - Folder with code for running Kalman filter
│   
└───line_segment - Folder with code for predicting lines, and running line-based classifier
│   
└───object_shadow - Folder with code for predicting object and shadow masks, and running object-shadow-based classifier
│   
└───perspective_fields - Folder with code for predicting perspective fields, and running perspective fields-based classifier
│   
└───results_extractor - Folder with code for generating test splits, and compiling result graphs
│   
└───sora_detector - Folder with prompts used for Sora Detector, and output predictions from GPT-4o using Perplexity
```

## Conda Environment
We use conda to manage packages for running different parts of the project. Unless otherwise stated, we run our code with the `projective-geometry` environment. Install the environment with `conda env create -f projective_geometry.yml`.

## Recommended Order of Running Project
1. Create and activate the `projective-geometry` conda environment
2. Create `./data/fake_video_paths.txt` and `./data/real_video_paths.txt` with lists of paths to videos to save frames from
3. Run `python frame_extraction.py` to save raw frames
4. Run `python keyframe_selector.py` to extract key frames from raw frames
5. Follow directions in `./object_shadow` to predict masks and run object-shadow-based classifier
6. Follow directions in `./perspective_fields` to predict fields and run fields-based classifier
7. Follow directions in `./line_segment` to predict lines and run lines-based classifier
8. Follow directions in `./kalman_filter` to run Kalman filter
9. Follow directions in `./baseline` to run baseline classifier
10. Follow directions in `./results_extractor` to extract difficulty-based test splits, and compile results
11. **Optional:** Prompt GPT-4o with Sora Detector prompts provided in `./sora_detector/prompts.txt` and/or look at our case study raw results in `./sora_detector/results`

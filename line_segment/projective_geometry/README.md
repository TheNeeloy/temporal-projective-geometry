## File Structure

```
./
│   README.md - This file
|   test.py - Inference script to run line-segment-based classifier
|   lines_model.py - Mode code
│
└───checkpoints - Folder with pretrained classifier checkpoint
│
└───results - Folder where classifier predictions are saved
```

Based on code from https://github.com/hanlinm2/projective-geometry/tree/main/line_segment

## Running Code
1. Create `./checkpoints` folder, download `Lines_combined.pt` from link above, and move it into the checkpoints folder
2. Update paths in `test.py`
3. Run `python test.py` or `python test.py --only_key_frames`

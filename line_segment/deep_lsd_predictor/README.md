## File Structure

```
./
│   README.md - This file
|   lines_extractor.py - Inference script to save line segment predictions
|   deep-lsd.yml - Conda environment file
```

Based on code from https://github.com/cvg/DeepLSD/blob/main/notebooks/demo_line_detection.ipynb

## Conda Environment
This code requires the `deep-lsd` environment. Install the environment with `conda env create -f deep-lsd.yml`.

## Running Code
1. Create and activate the `deep-lsd` conda environment
2. Update paths in `lines_extractor.py`
3. Run `python lines_extractor.py`

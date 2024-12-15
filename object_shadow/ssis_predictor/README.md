## File Structure

```
./
│   README.md - This file
|   object_shadow_extractor.py - Inference script to save mask predictions
|   predictor.py - Model code
|   ssis.yml - Conda environment file
```

Based on code from https://github.com/stevewongv/SSIS/tree/main/demo

## Conda Environment
This code requires the `ssis` environment. Install the environment with `conda env create -f ssis.yml`.

## Running Code
1. Create and activate the `ssis` conda environment
2. Update paths in `object_shadow_extractor.py` and `predictor.py`
3. Run `python object_shadow_extractor.py`

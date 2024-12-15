## File Structure

```
./
│   README.md - This file
|   fields_extractor.py - Inference script to save field predictions
|   perspective.yml - Conda environment file
```

Based on code from https://github.com/jinlinyi/PerspectiveFields/blob/main/demo/demo.py

## Conda Environment
This code requires the `perspective` environment. Install the environment with `conda env create -f perspective.yml`.

## Running Code
1. Create and activate the `perspective` conda environment
2. Update paths in `fields_extractor.py`
3. Run `python fields_extractor.py`

# Preprocessing (tile extraction)

This folder contains the preprocessing notebook used to extract tile-based datasets from
UAV orthomosaics and polygon annotations.

## Data
Raw UAV imagery, orthomosaics, and annotation files are not included in this repository
due to confidentiality and data protection constraints.

## Notebook
- `extract_data_blocks.ipynb`: Generates dataset versions under `OUT_ROOT/` in the form:
  `dataset_0000/High Risk/*` and `dataset_0000/No Risk/*` (png or npy)

## Usage (example)
Open the notebook and set the following paths:
- RGB orthomosaic path
- (Optional) NDVI raster path
- Polygon annotation path
- OUT_ROOT output folder

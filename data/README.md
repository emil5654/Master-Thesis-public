## Data

Raw UAV imagery, orthomosaics, and annotation files are **not included** in this repository
due to confidentiality and data protection constraints.

The preprocessing pipeline generates multiple tile-based dataset versions, each corresponding
to a specific preprocessing configuration (e.g. tile size, stride, or annotation threshold).

Expected structure (if data access is granted):

OUT_ROOT/
├── dataset_0000/
│   ├── High Risk/
│   │   └── *.png or *.npy
│   └── No Risk/
│       └── *.png or *.npy
├── dataset_0001/
│   ├── High Risk/
│   └── No Risk/
└── ...

Each dataset directory contains tile-level samples stored either as RGB images (`.png`)
or NumPy arrays (`.npy`). Class labels are encoded implicitly by the folder structure.

Dataset splits are defined using spatial block partitioning to prevent spatial leakage
between training, validation, and test sets.

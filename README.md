# Lidar Classification

Authors:
- Mikołaj Gałkowski
- Julia Przybytniowska
- Łukasz Tomaszewski
- Jakub Kubacki

----------------------
 
### Project setup

0. Install [uv](https://docs.astral.sh/uv/).
1. Sync with the project.

```bash
uv sync
```
2. Instal `pre-commit` [if not having in global python - `pip install pre-commit` or within `uv tool`].

```bash
uv tool install pre-commit
```

3. Install pre-commit hooks.

```bash
uvx pre-commit install
# (below equivalent to above, above one is more convenient)
uv tool run pre-commit install

```

### Contributing:

1. When adding new dependency:

```bash
uv add <package-name>
```

----------------------

### Project structure

```bash
lidar-classification/
│── data/                   # Raw & processed LiDAR data
│   ├── raw/                # Raw LiDAR point clouds (LAS/LAZ files)
│   │   └── ... 
│   ├── processed/          # Processed LiDAR data (normalized, clipped, etc.)
│   │   └── ...
│
│── notebooks/              # Jupyter Notebooks for exploration
│   └── 01_data_exploration.ipynb
│
│── src/                    # Source code
│   └── ... 
│
│── models/                 # Saved machine learning models
│   └── ... 
│
│── results/                # Classification results & evaluation
│   └── ... 
│
│── configs/                # Configuration files
│   └── ... 
│
│── scripts/                # Utility scripts for automation
│   └── ... 
│
│── tests/                  # Unit and integration tests
│   └── ...
│
│── .pre-commit-config.yaml # Pre-commit hooks (e.g., Ruff, uv)
│── pyproject.toml          # Python dependencies 
│── .gitignore              # Ignore unnecessary files
└── README.md               # Project description
```
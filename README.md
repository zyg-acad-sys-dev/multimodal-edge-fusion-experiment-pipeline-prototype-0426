# Multimodal Edge Fusion Experiment Pipeline Prototype

This repository contains preliminary experiment code for a multimodal edge-fusion research pipeline. The current version uses UTD-style action data as the first test case, with Skeleton and Inertial features as the core modalities and optional Depth/RGB extensions when available.

The code is not designed as a production system. It is a small research prototype for testing pipeline structure, feature extraction, clean-condition model comparison, and degraded-sensing robustness.

## Main Purpose

This prototype is used to organize and reproduce experiments involving:

- dataset scanning and sample indexing
- lightweight multimodal feature extraction
- clean-condition baseline comparison
- degraded-sensing robustness testing
- optional modality extension experiments
- result-table and figure export
- bilingual technical appendix documentation

The current implementation is tested on UTD-style action data, while the pipeline layout is kept modular so that it can be adapted to other multimodal datasets later.

## Repository Structure

```text
multimodal-edge-fusion-experiment-pipeline-prototype-0426/
│
├── code/
│   ├── config.py
│   ├── utils_io.py
│   ├── utils_features.py
│   ├── utils_models.py
│   ├── 00_check_env.py
│   ├── 01_scan_dataset.py
│   ├── 02_extract_features.py
│   ├── 03_train_clean_baselines.py
│   ├── 04_eval_degraded_sensing.py
│   ├── 05_export_tables_figures.py
│   ├── 06_optional_extract_depth_rgb_features.py
│   ├── 07_train_extended_modalities.py
│   ├── 08_eval_extended_degraded_robustness.py
│   ├── run_all_core.py
│   ├── run_extended.py
│   └── requirements.txt
│
├── docs/
│   ├── Technical_Appendix_EN.pdf
│   └── Technical_Appendix_CN.pdf
│
└── README.md

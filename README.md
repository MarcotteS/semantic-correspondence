# Semantic Correspondance with Visual Foundation Models

AML Course Project - Politecnico di Torino

## Project Description
Implementation of semantic correspondence methods using different visual foundation models : 
- **DINOv2**
- **DINOv3**
- **SAM (Segment Anything)**
- **SD (Stable Diffusion)**

Performance is measured on the **SPair-71k** benchmark, using the **PCK (Percentage of Correct Keypoints)** metric.

### Repo Structure
- `data/` indicates where to store the downloaded data from SPair-71k (also downloaded in the notebooks)
- `src/` regroups all the classes and methods used in this project (extractors for each model, evaluation methods...)
- `scripts/` contains mainly python scripts to run for evaluating the different feature extraction models
- `results\` stores the files produced by the execution of the scripts
- `notebooks/` contains different notebooks used to develop this project (with Colab)

### Requirements

### Download data and models

### Run the scripts

## References
* Tang et al., NeurIPS 2023 - Emergent Correspondence from Image Diffusion
* Zhang et al., NeurIPS 2023 - A Tale of Two Features: Stable Diffusion Complements DINO for
Zero-Shot Semantic Correspondence
* Min et al., ICCV 2019 - SPair-71k: A Large-scale Benchmark for Semantic Correspondence
* Oquab et al., CVPR 2023 - DINOv2: Learning Robust Visual Features without Supervision
* Simeoni et al., 2025 - DINOv3
* Kirillov et al., ICCV 2023 - Segment Anything (SAM)



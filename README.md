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
- `data/` where to store the downloaded data from SPair-71k (also downloaded in the notebooks)
- `src/` regroups all the classes and methods used in this project (extractors for each model, evaluation methods...)
- `scripts/` contains Python scripts to run for evaluating the different feature extraction models
- `backbones/` contains cloned model repositories (source code for DINOv3, etc.)
- `weights/` stores all model checkpoint files (`.pth` files for DINOv2, DINOv3, SAM, and fine-tuned models)
- `notebooks/` contains different notebooks used to develop this project (with Colab)


### Requirements

### Download data and models

### Run the scripts
- To evaluate the features extraction with DINOv2, DINOv3 and SAM (without fine-tuning): `./scripts/run_baseline.py`
- To run the fine-tuned models :
- To evaluate the features extraction with SD: `./scripts/run_sd.py`
- To evaluate the features extraction with the fusion of SD and DINOv2 : `./scripts/run_sddino_fusion.py`

## References
* Tang et al., NeurIPS 2023 - Emergent Correspondence from Image Diffusion
* Zhang et al., NeurIPS 2023 - A Tale of Two Features: Stable Diffusion Complements DINO for
Zero-Shot Semantic Correspondence
* Min et al., ICCV 2019 - SPair-71k: A Large-scale Benchmark for Semantic Correspondence
* Oquab et al., CVPR 2023 - DINOv2: Learning Robust Visual Features without Supervision
* Simeoni et al., 2025 - DINOv3
* Kirillov et al., ICCV 2023 - Segment Anything (SAM)



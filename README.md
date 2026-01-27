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
- `scripts/` contains some useful python scripts
- `notebooks/` contains different files including:
    - `training_free_baseline.ipynb` extracting features with the three backbones and evaluating the models with PCK metrics, as well as fine-tuning DINOv2
    - `task3_evaluation.ipynb` using window soft-argmax instead of simple argmax on the similarity map
    - `sd-1-5-baseline.ipynb` using a Stable Diffusion model (version 1.5) to extract features

## References


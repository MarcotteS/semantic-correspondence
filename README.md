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
- `weights/` place all model checkpoint files here (`.pth` files for DINOv3, SAM, and fine-tuned models)
- `notebooks/` contains different notebooks used to develop this project (with Colab)


### Requirements

```bash
pip install -r requirements.txt
```
### Download data and models

**SPair-71k Dataset:**
```bash
cd data/
wget https://cvlab.postech.ac.kr/research/SPair-71k/data/SPair-71k.tar.gz
tar -xzf SPair-71k.tar.gz
```

**Model weights** (place in `weights/`):
- **DINOv2**: Downloaded automatically via torch hub
- **DINOv3**: Download from [Meta AI](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/) → `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`
- **SAM**: Download from [segment-anything releases](https://github.com/facebookresearch/segment-anything#model-checkpoints) → `sam_vit_b_01ec64.pth`
- **Stable Diffusion v1.5**: Download from [Hugging Face](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) → place in `weights/sd-1-5-weights/`

**DINOv3 repository** (place in `backbones/`):
```bash
cd backbones/
git clone https://github.com/facebookresearch/dinov3.git
```

### Run the scripts

Each script contains configuration variables at the top of the file (model selection, batch size, image size, etc.) that can be modified before running.

- **Baseline evaluation** (DINOv2, DINOv3, SAM without fine-tuning): `python scripts/run_baseline.py`
- **Fine-tuned models evaluation** (Window Soft-Argmax): `python scripts/window_softargmax_eval.py`
- **Fine-tuning training**: `python scripts/fine_tuning.py`
- **Stable Diffusion evaluation**: `python scripts/run_sd.py`
- **SD + DINOv2 fusion**: `python scripts/run_sddino_fusion.py`

## References
* Tang et al., NeurIPS 2023 - Emergent Correspondence from Image Diffusion
* Zhang et al., NeurIPS 2023 - A Tale of Two Features: Stable Diffusion Complements DINO for
Zero-Shot Semantic Correspondence
* Min et al., ICCV 2019 - SPair-71k: A Large-scale Benchmark for Semantic Correspondence
* Oquab et al., CVPR 2023 - DINOv2: Learning Robust Visual Features without Supervision
* Simeoni et al., 2025 - DINOv3
* Kirillov et al., ICCV 2023 - Segment Anything (SAM)



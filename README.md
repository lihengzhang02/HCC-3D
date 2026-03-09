<p align="center">
<h1 align="center"><strong>HCC-3D: Hierarchical Compensatory Compression for 98% 3D Token Reduction in Vision-Language Models</strong></h1>
  <p align="center">
  Liheng Zhang<sup>1</sup>, Jin Wang<sup>1</sup>, Hui Li<sup>2</sup>, Bingfeng Zhang<sup>1†</sup>, Weifeng Liu<sup>1</sup>
  <br>
<sup>1</sup> China University of Petroleum (East China) &nbsp; <sup>2</sup> The Hong Kong Polytechnic University &nbsp; <sup>†</sup> Corresponding Author
  </p>
</p>
<p align="center">
    <a><strong>AAAI 2026</strong></a>
    <a href='https://github.com/lihengzhang02/HCC-3D'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
    <a href='https://arxiv.org/abs/2511.09883'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
    <a href='https://huggingface.co'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a>
</p>

## News

- **[2026/03]** 🚀 Code and model checkpoints are released.
- **[2026/01]** 🎉 HCC-3D is accepted by **AAAI 2026**.

## Highlights

- 🗜️ **98% token compression**: HCC-3D reduces 3D visual tokens from **513 → 12** — the most aggressive compression among all existing 3D-VLMs.
- 🏆 **State-of-the-art**: Outperforms MiniGPT-3D by **+1.04%** on ModelNet40 and **+1.00%** on Objaverse classification.
- ⚡ **52% faster training** and **20% faster inference** vs. MiniGPT-3D, running on a **single RTX 4090 (24G)** in just **11.9 hours**.
- 🔧 **Plug-and-play**: The HCC module is architecture-agnostic and generalizes across different 3D-VLM frameworks (demonstrated with GreenPLM).

## Method

![architecture](assets/architecture.png)

HCC-3D employs a **dual-path hierarchical compression** strategy:

- **Global Structure Compression (GSC):** `ng=8` learnable spatial queries attend over all 513 input tokens via multi-head cross-attention, producing a compact global representation that preserves overall 3D geometry.
- **Adaptive Detail Mining (ADM):** Identifies under-attended but semantically rich regions missed by GSC using a complementary scoring mechanism (attention coverage × MLP importance), selects the top-K=96 features, and recompresses them into 4 detail tokens — yielding **12 tokens total**.

## Results

### 3D Object Classification

| Model | Venue | LLM | #Tokens | ModelNet40 | Objaverse | Avg |
|-------|-------|-----|:-------:|:----------:|:---------:|:---:|
| PointLLM-7B | ECCV24 | 7B | 513 | 50.85 | 62.50 | 56.68 |
| ShapeLLM-13B | ECCV24 | 13B | 512 | 50.96 | 62.25 | 56.61 |
| MiniGPT-3D | MM24 | 2.7B | 513 | 61.24 | 66.75 | 64.00 |
| **HCC-3D (Ours)** | **AAAI26** | **2.7B** | **12** | **62.28** | **67.75** | **65.02** |

### 3D Object Captioning (Objaverse)

| Model | Venue | #Tokens | Qwen2 | Sentence-BERT | SimCSE | Avg |
|-------|-------|:-------:|:-----:|:-------------:|:------:|:---:|
| MiniGPT-3D | MM24 | 513 | 48.17 | 49.54 | 51.39 | 49.70 |
| **HCC-3D (Ours)** | **AAAI26** | **12** | **48.72** | **50.89** | 50.84 | **50.15** |

### Efficiency

| Method | GPU | Training Time | Inference Speed |
|--------|-----|:---:|:---:|
| PointLLM-13B | 8×A100 (80G) | 213h | ~3.45s/sample |
| ShapeLLM-13B | 8×A800 (80G) | 160h | ~2.04s/sample |
| MiniGPT-3D | 1×RTX4090 (24G) | 16.8h | 0.45s/sample |
| **HCC-3D** | **1×RTX4090 (24G)** | **11.9h** | **0.36s/sample** |

## Getting Started

### Installation

```bash
git clone https://github.com/lihengzhang02/HCC-3D.git
cd HCC-3D
conda env create -f environment.yml
conda activate hcc_3d
bash env_install.sh
```

Tested environment: 1× RTX 4090 24GB / Ubuntu 20.04 / CUDA 11.8 / Python 3.9 / PyTorch 2.0.0

### Data Preparation

Download all data files (~78GB) from [PointLLM HuggingFace Datasets](https://huggingface.co/datasets/RunsenXu/PointLLM/tree/main).

```bash
# Merge and extract Objaverse point clouds
cat Objaverse_660K_8192_npy_split_a* > Objaverse_660K_8192_npy.tar.gz
tar -xvf Objaverse_660K_8192_npy.tar.gz
```

Organize as follows:

```
HCC-3D/data
├── anno_data/
│   ├── PointLLM_brief_description_660K.json
│   ├── PointLLM_brief_description_660K_filtered.json
│   ├── PointLLM_brief_description_val_200_GT.json
│   ├── PointLLM_complex_instruction_70K.json
│   ├── object_ids_660K.txt
│   └── val_object_ids_3000.txt
├── modelnet40_data/
│   └── modelnet40_test_8192pts_fps.dat
└── objaverse_data/
    ├── 00000054c36d44a2a483bdbff31d8edf_8192.npy
    └── ...
```

### Model Weights

Download from [HuggingFace](https://huggingface.co/lihengzhang02/HCC-3D) and place under `params_weight/`:

```
params_weight/
├── HCC_3D_stage_3/           # Stage III checkpoint
├── HCC_3D_stage_4/           # Stage IV checkpoint
├── Phi_2/                    # LLM backbone (Phi-2, 2.7B)
├── pc_encoder/               # Point-BERT encoder
├── all-mpnet-base-v2/        # Captioning evaluation
└── sup-simcse-roberta-large/ # Captioning evaluation
```

## Training

```bash
export PYTHONPATH=$PWD

# Stage I:
CUDA_VISIBLE_DEVICES=0 python train.py --cfg-path ./train_configs/HCC_3D/stage_1.yaml

# Stage II:
CUDA_VISIBLE_DEVICES=0 python train.py --cfg-path ./train_configs/HCC_3D/stage_2.yaml

# Stage III:
CUDA_VISIBLE_DEVICES=0 python train.py --cfg-path ./train_configs/HCC_3D/stage_3.yaml

# Stage IV:
CUDA_VISIBLE_DEVICES=0 python train.py --cfg-path ./train_configs/HCC_3D/stage_4.yaml
```

## Evaluation

### Step 1 — Generate outputs

```bash
export PYTHONPATH=$PWD

# ModelNet40 classification (run for prompt_index 0 and 1)
CUDA_VISIBLE_DEVICES=0 python hcc3d/eval/eval_modelnet.py \
    --out_path ./output/test \
    --cfg-path ./eval_configs/benchmark_evaluation_paper.yaml \
    --prompt_index 0

# Objaverse classification (run for prompt_index 0 and 1)
CUDA_VISIBLE_DEVICES=0 python hcc3d/eval/eval_objaverse.py \
    --out_path ./output/test --task_type classification \
    --cfg-path ./eval_configs/benchmark_evaluation_paper.yaml \
    --prompt_index 0

# Objaverse captioning
CUDA_VISIBLE_DEVICES=0 python hcc3d/eval/eval_objaverse.py \
    --out_path ./output/test --task_type captioning \
    --cfg-path ./eval_configs/benchmark_evaluation_paper.yaml \
    --prompt_index 2
```

### Step 2 — LLM-based evaluation with Qwen2-72B-Instruct (Recommended)

We follow [GreenPLM](https://arxiv.org/abs/2408.15966) and use **Qwen2-72B-Instruct** as the evaluator for cost-effective and reproducible results. Get your API key from [Alibaba Cloud](https://bailian.console.aliyun.com/#/api-key), or self-host locally following the [Qwen2 repo](https://github.com/QwenLM/Qwen2).

```bash
export PYTHONPATH=$PWD
export DASHSCOPE_API_KEY=sk-xxx

# Objaverse classification
python hcc3d/eval/evaluator_qwen.py \
    --results_path ./output/test/Objaverse_classification_prompt0.json \
    --eval_type open-free-form-classification \
    --model_type qwen2-72b-instruct --parallel --num_workers 4

# ModelNet40 classification
python hcc3d/eval/evaluator_qwen.py \
    --results_path ./output/test/ModelNet_classification_prompt0.json \
    --eval_type modelnet-close-set-classification \
    --model_type qwen2-72b-instruct --parallel --num_workers 4

# Captioning
python hcc3d/eval/evaluator_qwen.py \
    --results_path ./output/test/Objaverse_captioning_prompt2.json \
    --eval_type object-captioning \
    --model_type qwen2-72b-instruct --parallel --num_workers 4
```

### Step 3 — Traditional metrics (Sentence-BERT & SimCSE)

```bash
CUDA_VISIBLE_DEVICES=0 python hcc3d/eval/traditional_evaluator.py \
    --results_path ./output/test/Objaverse_captioning_prompt2.json
```
## Demo

```bash
python demo.py --cfg-path ./eval_configs/HCC_3D_demo.yaml --gpu-id 0
```

## TODO

- [x] Release training code
- [x] Release evaluation code
- [x] Release model checkpoints
- [ ] Release Gradio demo
- [ ] HuggingFace Spaces online demo

## Citation

```bibtex
@inproceedings{zhang2026hcc3d,
  title     = {HCC-3D: Hierarchical Compensatory Compression for 98\% 3D Token Reduction in Vision-Language Models},
  author    = {Zhang, Liheng and Wang, Jin and Li, Hui and Zhang, Bingfeng and Liu, Weifeng},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2026}
}
```

## Related Work

- [LLaVA-Mini](https://github.com/ictnlp/LLaVA-Mini): Extreme 2D vision token reduction (576 → 1) for image and video VLMs
- [GreenPLM](https://github.com/TangYuan96/GreenPLM): 3D data-efficient point-language understanding
- [MiniGPT-3D](https://github.com/TangYuan96/MiniGPT-3D): Efficient 3D-LLM alignment with 2D priors
- [PointLLM](https://github.com/OpenRobotLab/PointLLM): Empowering LLMs to understand point clouds
- [ShapeLLM](https://arxiv.org/abs/2402.17766): 3D object understanding for embodied interaction

## Acknowledgements

We thank the authors of [PointLLM](https://github.com/OpenRobotLab/PointLLM), [MiniGPT-3D](https://github.com/TangYuan96/MiniGPT-3D), [GreenPLM](https://github.com/TangYuan96/GreenPLM), and [Point-BERT](https://github.com/lulutang0608/Point-BERT) for their excellent open-source work that this project builds upon.

## License

This project is released under the [CC BY-NC 4.0 License](LICENSE).

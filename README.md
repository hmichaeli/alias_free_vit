# Alias-Free Transformer: Fractional Shift Invariance via Linear Attention

Official PyTorch implementation of the NeurIPS 2025 paper **Alias-Free Transformer**. This repository extends the Cross-Covariance Image Transformer (XCiT) codebase with alias-free modifications.

## Overview

AFT introduces alias-free components throughout the vision transformer pipeline:

- Alias-free convolutional patch embedding with circular padding and controllable activations.
- UpGELU activation family for patch embedding, MLP, and local positional interactions.
- Adversarial shift evaluation suites for benchmarking spatial robustness.

All experiments reported in the paper were reproduced with this codebase using PyTorch 2.6 and `torchrun`.

## Repository highlights

- `main.py`: end-to-end training & evaluation entry point, including AFT-specific arguments and shift evaluation tasks.
- `af_ops.py`: alias-free convolution layers, Upsample AF blocks, and helpers.
- `xcit.py`, `xcit_af.py`, `xcit_aps.py`: model definitions for Baseline, Alias-Free (AF) and Adaptive polyphase sampling (APS) models.
- `adversarial_shift_eval.py` / `shift_eval.py`: shift-consistency and adversarial robustness benchmarks.
- `TRAINING_COMMANDS.md`:  training commands for Nano/Small, baseline/APS/AFT variants.

Experiments were run with **PyTorch 2.6** using `torchrun` for distributed training. We recommend Python 3.10+ and CUDA 12.x.

## Data preparation

All experiments were conducted on ImageNet-1k. After extracting the dataset, point `--data-path` at the directory containing `train/` and `val/`. Custom datasets can be integrated by following the interface implemented in `datasets.py`.

## Training

We recommend launching training with `torchrun` (single node, multi-GPU):

```bash
torchrun --nproc_per_node=8 --master_port=1534 main.py \
    --model xcit_af_small_12_p16 \
    --use-pos false \
    --conv-padding-mode circular \
    --pe-down-af true \
    --pe-fuse-down true \
    --pe-act up_gelu \
    --xca-norm-layer layer_af \
    --mlp-act up_gelu \
    --lpi-act up_gelu \
    --features-type cls_attn_af \
    --drop-path 0.05 \
    --batch-size 64 \
    --output_dir=$OUT_DIR \
    --data-path $DATA_PATH
```

For a full list of reproduction commands (Nano / Small, baseline / APS / AFT), see [`TRAINING_COMMANDS.md`](./TRAINING_COMMANDS.md).

### Logging & checkpoints

- Enable Weights & Biases logging with `--wandb_project <name>` (optional).
- Checkpoints are written to `--output_dir`. Use `--save-every-epoch` to create periodic snapshots.
- Resume training by pointing `--resume` to an existing `checkpoint.pth`.

## Evaluation

### Classification

```bash
python main.py --eval --model xcit_af_small_12_p16 --data-path $DATA_PATH --pretrained $CKPT --output_dir=$OUT_DIR
```

### Shift robustness

`main.py` includes three alias-free robustness tasks:

- `--task adversarial_crop_shift`
- `--task adversarial_cyclic_shift`
- `--task adversarial_bilinear_fractional_shift`

Adjust `--max-shift`, `--crop-input-size`, and `--upsample` to explore different regimes. See `adversarial_shift_eval.py` for metric definitions.

## Repository structure

```
alias_free_vit/
  ├── af_ops.py
  ├── adversarial_shift_eval.py
  ├── datasets.py
  ├── engine.py
  ├── losses.py
  ├── main.py
  ├── samplers.py
  ├── shift_eval.py
  ├── utils.py
  ├── xcit_*.py
  └── TRAINING_COMMANDS.md
```

## Citation

Please cite the paper if you use this repository:

```
@article{michaeli2025aft,
  title={Alias-Free ViT: Fractional Shift Invariance via Linear Attention},
  author={Michaeli, Hagay and Soudry, Daniel},
  journal={Advances in Neural Information Processing Systems},
  year={2025}
}
```

## Acknowledgements

This implementation builds upon the [XCiT codebase](https://github.com/facebookresearch/xcit). We thank the original authors for releasing their code under a permissive license. Alias-free modifications are © 2025 Hagay Michaeli.

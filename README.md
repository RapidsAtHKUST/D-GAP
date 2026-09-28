# README

Source code of paper 'D-GAP: Improving Out-of-Domain Robustness via Dataset-Agnostic and Gradient-Guided Augmentation in Amplitude and Pixel Spaces'.

---

## Abstract

Out-of-domain (OOD) robustness is challenging to achieve in real-world computer vision, especially in unsupervised domain adaptation scenarios, where shifts in image background, style, and acquisition instruments often degrade model performance. Generic augmentations show inconsistent gains under such shifts, whereas dataset-specific augmentations require expert knowledge and prior analysis. Moreover, prior studies show that neural networks adapt poorly to domain shifts because they exhibit a learning bias to domain-specific frequency components. Perturbing frequency values can mitigate such bias but overlooks pixel-level details, leading to suboptimal performance. To address these limitations, we propose D-GAP, a Dataset-agnostic and Gradient-guided augmentation method for the Amplitude spectrum (in frequency space) and the Pixel values. Unlike conventional handcrafted augmentations, D-GAP computes sensitivity maps in the frequency space from task gradients, which reflect how strongly the deep models respond to different frequency components, and uses the maps to adaptively interpolate amplitudes between source and target samples. We further propose a dual-space augmentation that jointly controls spectral bias and spatial fidelity by introducing a complementary pixel-space blending branch. This way, D-GAP turns augmentation from fixed, random, or manually designed perturbation into a model-response-adaptive intervention. Extensive experimental results show that the proposed method consistently outperforms both generic and dataset-specific domain adaptation methods, improving average OOD performance by +5.3% on four real-world datasets and +1.9% on three benchmark datasets.

---

## Run Examples for iWildCam, Camelyon17 and Birdcalls:

```bash
#!/bin/bash
SEED=42
LP_LR=0.006222466404167087
FT_LR=0.003324366874654924
TRANSFORM_P=0.8260829829811762
AUG_STRENGTH=0.0995477425665205
PRETRAIN_PATH=ckp-55.pth

# load the environment
source env_setup.sh

DATA_DIR=/connect-later-wilds
LOGDIR=/connect-later-wilds/logs_cam

mkdir -p $LOGDIR

SUFFIX=4

# LP
python examples/run_expt.py --root_dir ${DATA_DIR} \
    --lr ${LP_LR} \
    --n_epochs 10 \
    --weight_decay 0.01 \
    --transform_p ${TRANSFORM_P} \
    --algorithm ERM \
    --dataset camelyon17 \
    --download \
    --pretrained_model_path ${PRETRAIN_PATH} \
    --seed ${SEED} \
    --log_dir ${LOGDIR}/LP_camelyon_pretrain${SUFFIX}_aug_${SEED}_lplr${LP_LR}_ftlr${FT_LR}_p${TRANSFORM_P}_augstrength${AUG_STRENGTH} \
    --progress_bar True \
    --erm_freeze_featurizer

# FT
python examples/run_expt.py --root_dir ${DATA_DIR} \
    --lr ${FT_LR} \
    --weight_decay 0.01 \
    --transform_p ${TRANSFORM_P} \
    --algorithm ERM \
    --dataset camelyon17 \
    --download \
    --pretrained_model_path ${LOGDIR}/LP_camelyon_pretrain${SUFFIX}_aug_${SEED}_lplr${LP_LR}_ftlr${FT_LR}_p${TRANSFORM_P}_augstrength${AUG_STRENGTH}/camelyon17_seed:${SEED}_epoch:best_model.pth \
    --seed ${SEED} \
    --log_dir ${LOGDIR}/FT_camelyon_pretrain${SUFFIX}_aug_LPFT_${SEED}_lplr${LP_LR}_ftlr${FT_LR}_p${TRANSFORM_P}_augstrength${AUG_STRENGTH} \
    --progress_bar True \
```

## Run Example for Galaxy10:

```bash
python examples/galaxy10.py
```


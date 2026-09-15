# A diffusion model of viral evolution predicts mutation fitness and evolutionary trajectories

> Viral Evolution Simulator (VES), a diffusion model-based framework that mirrors the duality of viral evolution by design: forward noise injection simulates stochastic mutation, and reverse denoising recapitulates selective filtering.

<p align="center">
  <img src="assets/framework.png" width="80%" alt="Overall Framework">
  <br>
  <em> Modelling viral evolution via diffusion models </em>
</p>

## 📌 Abstract

Viral evolution arises from random mutations and natural selection, yet computational approaches rarely model these two forces in a unified way. We present Viral Evolution Simulator (VES), a diffusion model-based framework that mirrors this duality by design: forward noise injection simulates stochastic mutation, and reverse denoising recapitulates selective filtering. Trained solely on viral protein sequences, VES predicts mutational fitness without functional data, measuring fitness as the reconstruction difficulty of a mutated sequence relative to its wild-type counterpart. Across immune escape, receptor binding, and deep mutational scanning datasets, VES outperforms state-of-the-art generative models, achieving a 31.78% error reduction over the best baseline in immune escape mutation fitting evaluation. When trained on sequences collected before June 2024 and evaluated against H1N1 strains that later emerged, VES assigned high scores to 16 of 20 mutations that subsequently showed the sharpest frequency shifts. Extending to avian influenza H5, the framework reveals a dynamic interplay between antigenic escape and human-type receptor binding. Both functions dropped sharply in 2021, followed by a sustained rise in receptor affinity that could connect to recent epidemiological trends. VES offers a generalizable, sequence-only foundation for tracing evolutionary trajectories and prioritizing mutations for surveillance and experimental validation, pointing toward where functional efforts might matter most.

## 🛠️ Requirements
torch 2.8.0+cu128
numpy==2.3.2
pandas==2.3.3
scikit-learn==1.7.2
scipy==1.17.1
biopython==1.85
biotite==1.6.0
matplotlib==3.10.5
seaborn==0.13.2
logomaker==0.8.7
tqdm>=4.66
tensorboard==2.20.0
pyyaml==6.0.2

## 📂 Datasets
### Training Data
- **Source**：GISAID
- **Volume**：95,560 sequences for influenza virus (hemagglutinin protein); 55,246 sequences for SARS-CoV-2 viruses (spike protein)
- **Pre-processing**：quality control and 100% redundancy removal
- **Alignment**：multiple sequence alignment using MAFFT, selecting the Influenza A virus (A/WSN/1933(H1N1)) strain and the hCoV-19/Wuhan/WIV04/2019 (WIV04) strain of SARS-CoV-2 viruses as the wild type reference sequences for the respective alignments

### Evaluation Data
- **Source**：
[1] How single mutations affect viral escape from broad and narrow antibodies to H1 influenza hemagglutinin
[2] Accurate Measurement of the Effects of All Amino-Acid Mutations on Influenza Hemagglutinin
[3] Complete mapping of viral escape from neutralizing antibodies
[4] Deep mutational scanning of H5 hemagglutinin to inform influenza virus surveillance

## 🚀 Training
| Parameter | Value |
|------|------|
| Batch size | 1024 |
| Learning rate | 1e-4 |
| Epochs | 60 |
| Optimizer | AdamW |
| Hardware | single NVIDIA GeForce RTX 5090 GPU |


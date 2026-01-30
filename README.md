# Automated Weld Seam Detection System Based on the Fusion of RGB-D Images
> **Authors:**
> [*Zijian Wu*]
> [*Ziqiang Wang*]


## 1. Preface

- This repository provides code for "_**Automated Weld Seam Detection System Based on the Fusion of RGB-D Images**_" 

## 2. Overview

### 2.1. Introduction

Automatic weld seam detection is a cornerstone of intelligent manufacturing, yet its advancement is severely impeded by the scarcity of high-quality 3D annotated data and the complex topology of weldments in cluttered industrial environments. To address the data acquisition bottleneck, we construct the first large-scale RGB-D weld seam detection benchmark. By leveraging 3D printing technology to synthesize large-tonnage marine weldment prototypes and employing stereo structured-light measurement for high-fidelity 3D reconstruction, this dataset bridges the gap between theoretical models and physical constraints. Methodologically, we propose a novel geometry-aware framework tailored for the thin, elongated morphology of weld seams. Specifically, a Depth Guidance Cross-modal Attention Module (DGCA) is designed to utilize depth maps as spatial attention priors, effectively suppressing background noise at the encoding stage. Complementarily, a Graph-based Cross-modal Enhancement Module (GCE) introduces graph-based interactions to model non-local geometric dependencies, thereby refining structural reasoning during decoding. Extensive experiments demonstrate that our approach significantly outperforms state-of-the-art general RGB-D segmentation methods. Notably, it exhibits superior robustness against environmental interference and demonstrates remarkable label efficiency in data-constrained regimes, offering a scalable solution for real-world industrial deployment.


### 2.2. Framework Overview

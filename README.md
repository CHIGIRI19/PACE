<<<<<<< HEAD
<h1 align="center">PACE</h1>

Follow the steps below to set up and run PACE locally.

1. Clone this repository


2. Create a new conda environment (PACE) using

```vim
conda create -n PACE python=3.10 ipykernel -y
conda activate PACE
```

3. Install requirements

```vim
cd PACE
pip install -r requirements.txt
```

## Datasets

All datasets used in this project are publicly available. The files in the `/data` directory have been uniformly processed and converted to CSV format for consistency and ease of use.

<!-- For the 2000-dimensional pancreas dataset:

- **Source**: Download `GSE132188_adata.h5ad.h5` from [NCBI GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE132188)
- **Preprocessing**: Follow the data processing pipeline in `notebooks/preprocess_pancreas.ipynb` to reproduce the formatted dataset. -->

## Tutorials

We currently provide the following tutorials, which can be easily followed in the `notebooks` directory:

- IPSCs data
<!-- - Dyngen
- Gaussian 1000D
- Mouse hematopoiesis
- Hold-out experiments for EB 5D, CITE 5D, and CITE 50D
- EB 50D
- Pancreas 2000D dataset (gene space) and deeper analysis -->

The notebook demonstrate how to reproduce our main experiments and can serve as practical starting points for applying PACE to new datasets.

<!-- ## Contact

If you encounter any problems or have questions about PACE, feel free to reach out to **Dongyi Wang** at [dongyiwang@stu.xjtu.edu.cn](mailto:dongyiwang@stu.xjtu.edu.cn).

## Citation

If you find PACE helpful for your research, please consider citing our work:

```bibtex
@inproceedings{
wang2025joint,
title={Joint Velocity-Growth Flow Matching for Single-Cell Dynamics Modeling},
author={Dongyi Wang and Yuanwei Jiang and Zhenyi Zhang and Xiang Gu and Peijie Zhou and Jian Sun},
booktitle={NeurIPS},
year={2025}
}
```

## Acknowledgments

This project has greatly benefited from the following open-source repositories. We extend our sincere gratitude to the authors and contributors for sharing their valuable work:

- [DeepRUOT](https://github.com/zhenyiizhang/DeepRUOT)
- [conditional-flow-matching](https://github.com/atong01/conditional-flow-matching)
- [MIOFlow](https://github.com/KrishnaswamyLab/MIOFlow) -->
=======
# PACE
>>>>>>> 700794216017e2c59db845b161438b407cdb3fd1

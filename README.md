# CtFMCRecon
This repository includes the implementation of our paper "Unsupervised Adaptive Implicit Neural Representation Learning for Scan-Specific MRI Reconstruction", which is developed based on the implementation of [NeRP](https://github.com/liyues/NeRP).

# Installation
The implementation is developed based on the following Python packages/versions, including:
```
python: 3.8.12
torch: 2.0.0
numpy: 1.20.3
tensorflow: 2.5.0
sigpy: 0.1.23
scipy: 1.10.1
```

# Running Code
The implementation for 2D and 3D multi-contrast reconstruction can be found in `train_CTF_2D.py` and `train_CTF_3D.py`. An example command as follows can be executed to run experiments using our proposed framework:
```
python train_CTF_2D.py --multi-contrast --contrast T2 --alpha 0.1 --un_rate 4 --steps 3
```
In this example, our framework is optimised using multi-contrast data (T1 as reference contrast), 4-fold under-sampling, the degree of assistance from the reference contrast as 0.1, and the k-space data divided into three portions.

## Training with Baselines
CS-MRI methods are considered as baseline methods in our papers, with total variation and wavelet transform applied separately as the regularsation terms. The implementation is based on the [BART](https://mrirecon.github.io/bart/) toolbox, and an installation is required to run the baseline methods on the dataset. The scripts for running the experiments can be found in `bart_2d.py` and `bart_3d.py`.  An example command as follows can be performed to learn to reconstruct:
```
python bart_2d.py --contrast T2 --un_rate 4 --reg wav --alpha 0.1
```
In this example, the CS-MRI method is performed on the dataset with 4-fold under-sampled T2w data to be reconstructed, with the wavelet tranform applied as the regularsation, using the regularsation strength of 0.1. The similar procedure can be followed for 3D reconstruction.

Please note that, the under-sampling pattern needs to be saved as `npy` file for the evaluation scripts to lead by using the function `get_vd_mask` in `common_masks.py`, ensuring the consistency.

## Evaluation
The evaluation scripts for both our proposed framework and the baseline methods are included in `eval_CS_2D.py`, `eval_CS_3D.py`, `eval_MR_2D.py`, and `eval_MR_2D.py`. With the provided paths to the reconstructed data and the associated parameters directly specified in the files, the evaluation can be performed to report the metrics. 


## Citation
If this implementation is helpful for your research, please consider citing our work:
```
@article{yang2023unsupervised,
  title={Unsupervised Adaptive Implicit Neural Representation Learning for Scan-Specific MRI Reconstruction},
  author={Yang, Junwei and Li{\`o}, Pietro},
  journal={arXiv preprint arXiv:2312.00677},
  year={2023}
}
```

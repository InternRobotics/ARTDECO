# ARTDECO: Towards Efficient and High-Fidelity On-the-Fly 3D Reconstruction with Structured Scene Representation



[![Project Website](https://img.shields.io/badge/ARTDECO-Website-4CAF50?logo=googlechrome&logoColor=white)](https://city-super.github.io/artdeco/)

[![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=b31b1b)](https://arxiv.org/abs/2510.08551)



[Guanghao Li*](https://lightingooo.github.io/), [Kerui Ren*](https://cskrren.github.io/), [Linning Xu](https://eveneveno.github.io/lnxu/),

[Zhewen Zheng](https://github.com/QuantumEPR), [Changjian Jiang](https://scholar.google.com/citations?hl=en&user=V4miywEAAAAJ), [Xin Gao](https://gaoxin492.github.io/), [Bo Dai](https://daibo.info/), [Jian Pu<sup>†</sup>](https://scholar.google.com/citations?user=9pUCoOkAAAAJ&hl=en), [Mulin Yu<sup>†</sup>](https://mulinyu.github.io/), [Jiangmiao Pang](https://oceanpang.github.io/) <br/>
## 📢 News

**[2026.02.19]** 🚀 Training code is officially released!
**[2026.01.26]** 🎉 Our paper has been accepted by **ICLR 2026**.

---

## 🏗️ System Architecture
### Frontend and Backend Modules
![img](assets/pipeline1.png)
(a) Frontend: Images are captured from the scene and streamed into the front-end part. Each incoming frame is aligned with the latest keyframe using a matching module to compute pixel correspondences. Based on the correspondence ratio and pixel displacement, the frame is classified as a keyframe, a mapper frame, or a common frame. The selected frame, along with its pose and point cloud, is then passed to the back-end. (b) Backend: For each new keyframe, a loop-detection module evaluates its similarity with previous keyframes. If a loop is detected, the most relevant candidates are refined and connected in the factor graph; otherwise, the keyframe is linked only to recent frames. Finally, global pose optimization is performed with Gauss–Newton, and other frames are adjusted accordingly. We instantiate the matching module with MASt3R and the loop-detection module with Pi3.

### Mapping Module
![img](assets/pipeline2.png)
When a keyframe or mapper frame arrives from the backend, new Gaussians are added to the scene. Multi-resolution inputs are analyzed with the Laplacian of Gaussian (LoG) operator to identify regions that require refinement, and new Gaussians are initialized at the corresponding monocular depth positions in the current view. Common frames are not used to add Gaussians but contribute through gradient-based refinement. Each primitive stores position, spherical harmonics (SH), base scale, opacity, local feature, dmax, and voxel index vid. For rendering, the dmax attribute determines whether a Gaussian is included at a given viewing distance, enabling consistent level-of-detail control.

## 🛠️ Installation

### Environment Setup

Our framework is validated on **Python 3.11/3.12**, **PyTorch 2.3.1/2.7.1**, and **CUDA 12.1/12.8**, generally compatible with recent PyTorch/CUDA releases.

1. Clone the repo.
```bash
git clone https://github.com/InternRobotics/ARTDECO.git
cd ARTDECO/
```

2. Create the environment and install PyTorch.
```bash
# python 3.11 + cuda 12.1 + pytorch 2.5.1
conda create -n artdeco python=3.11
conda activate artdeco
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```
or
```bash
# python 3.12 + cuda 12.8 + pytorch 2.7.1
conda create -n artdeco python=3.12
conda activate artdeco
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

### Dependency & Model Preparation

1. Download Checkpoints.

Place the required MASt3R and [Pi3 Checkpoints](https://huggingface.co/yyfz233/Pi3/resolve/main/model.safetensors?download=true) in the `models/` directory.
```
# Download MASt3R checkpoints
mkdir -p models/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P models/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth -P models/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl -P models/
```


1. Build VSLAM Module.
```bash
# Install VSLAM thirdparty
cd VSLAM
pip install -e thirdparty/mast3r --no-build-isolation
pip install -e thirdparty/in3d --no-build-isolation
pip install -e . --no-build-isolation

# Install Pypose and GeoCalib
pip install pypose
python -m pip install -e "git+https://github.com/cvg/GeoCalib#egg=geocalib"

cd ..
```

3. Build Reconstruct Module.
```bash
# Install gsplat
pip install gsplat

# Install submodules
cd Reconstruct
pip install submodules/fused-ssim --no-build-isolation
pip install submodules/simple-knn --no-build-isolation
pip install submodules/graphdecoviewer

cd ..
```

4. Install the remaining dependencies.

## 🚀 Quick Start (Training)

We provide the [PINGPONG](https://drive.google.com/file/d/1sllaCqCLt5mS0ZfDNY_tb-CZ3ruMhU2P/view?usp=sharing) dataset as a benchmark example.

### Data Structure

Organize your dataset as follows:

```text
<dataset_root>
└── pingpong
    ├── images
    │   └── ${timestamp}.png/.jpg
    └── (intr.yaml)
```
The reference `intr.yaml` is shown below:
```text
width: 2592
height: 1944
# fx, fy, cx, cy ...
calibration:  [1478.95393660578, 1478.95393660578, 1296.0, 972.0]
```
### Run Reconstruction

Execute the following command to start the on-the-fly reconstruction:

```bash
bash run.sh
```

---

## ✉️ Contact & Citation

For questions, please contact **Kerui Ren** ([renkerui@sjtu.edu.cn](mailto:renkerui@sjtu.edu.cn)).

If you find ARTDECO helpful for your research, please cite our work:

```bibtex
@article{li2025artdeco,
  title={Artdeco: Towards efficient and high-fidelity on-the-fly 3d reconstruction with structured scene representation},
  author={Li, Guanghao and Ren, Kerui and Xu, Linning and Zheng, Zhewen and Jiang, Changjian and Gao, Xin and Dai, Bo and Pu, Jian and Yu, Mulin and Pang, Jiangmiao},
  journal={arXiv preprint arXiv:2510.08551},
  year={2025}
}
```
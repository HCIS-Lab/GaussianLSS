# <div align="center">**GaussianLSS - Toward Real-world BEV Perception: Depth Uncertainty Estimation via Gaussian Splatting**</div>

This is the official repository of CVPR'25 paper: GaussianLSS - Toward Real-world BEV Perception: Depth Uncertainty Estimation via Gaussian Splatting.

Official implementation comming soon. Stay tuned!

# Installation
Create the environment with conda:
```bash
# Clone repo first
git clone https://github.com/HCIS-Lab/GaussianLSS

# Create python 3.8
conda create -y --name GaussianLSS python=3.8.0
conda activate GaussianLSS

# Install pytorch 2.1.0
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt

# Install Gaussian Splatting
# Check if your cudatoolkit's version is same as pytorch build.
cd GaussianLSS/model/diff-gaussian-rasterization
pip install -e .
```
# Dataset preparation
## nuScenes Dataset
Go to [nuScenes](https://www.nuscenes.org/nuscenes) and download & unzip the following data:
- Trainval
- Map expansion

After unzipping, create a link file:
```bash
mkdir data
ln -s {YOUR_NUSC_DATA_PATH} ./data/nuscenes
```

## Generate lable
Generate required labels for running via:
```bash
python scripts/generate_data.py
```
This packs 3D bounding boxes into individual files with ego poses. It would take within 10 minutes.

# Running
## Vehicle & Pedestrian
## Map(topology)

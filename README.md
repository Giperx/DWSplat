# Demo of DWSplat

## Installation
Our code relies on Python 3.10+, and is developed based on PyTorch 2.2.0 and CUDA 12.1, but it should work with other Pytorch/CUDA versions as well.

1. Clone DWSplat.
```bash
git clone https://github.com/Giperx/DWSplat.git
cd DWSplat
```

2. Create the environment, here we show an example using conda.
```bash
conda create -y -n dwsplat python=3.10
conda activate dwsplat
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
pip install torch-scatter==2.1.2+pt22cu121 -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
# install pytorch3d
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
pip install . --no-build-isolation
# install gsplat
wget "https://github.com/nerfstudio-project/gsplat/releases/download/v1.4.0/gsplat-1.4.0%2Bpt22cu121-cp310-cp310-linux_x86_64.whl"
pip install ./gsplat-1.4.0+pt22cu121-cp310-cp310-linux_x86_64.whl
# install for OmniVGGT
pip install trimesh onnxruntime-gpu==1.17.1 viser==0.2.23 evo -i https://pypi.tuna.tsinghua.edu.cn/simple
# install curope
cd src/model/encoder/backbone/croco/curope
python setup.py build_ext --inplace
cd ../../../../../../
```


## Training

```
# single node:
CUDA_VISIBLE_DEVICES=0,1 python src/main.py +experiment=nuscenes trainer.num_nodes=1
```



## Acknowledgement

We thank all authors behind these repositories for their excellent work: [AnySplat](https://github.com/InternRobotics/AnySplat), [OmniVGGT](https://github.com/Livioni/OmniVGGT-official), [VGGT](https://github.com/facebookresearch/vggt), [NoPoSplat](https://github.com/cvg/NoPoSplat), [CUT3R](https://github.com/CUT3R/CUT3R/tree/main) and [gsplat](https://github.com/nerfstudio-project/gsplat).

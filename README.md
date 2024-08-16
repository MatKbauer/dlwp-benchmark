# Deep Learning Weather Prediction Model and Backbone Comparison on Navier-Stokes and WeatherBench

A benchmark to compare different deep learning models and their backbones on synthetic Navier-Stokes and real-world data from [WeatherBench](https://arxiv.org/abs/2002.00469).

## Getting Started

To install the package, first create an environment, cd into it, and install the DLWPBench package via

```
conda create -n dlwpbench python=3.11 -y && conda activate dlwpbench
pip install -e .
```

In the pip Neuraloperator package, the tucker decomposition for TFNO is not installed, so manually install the package from the [source repository](https://github.com/NeuralOperator/neuraloperator) with

```
mkdir packages
cd packages
git clone https://github.com/NeuralOperator/neuraloperator
git checkout 05c01c3  # (optional) use the repository state that is compatible with checkpoints from our work
cd neuraloperator
pip install -e .
pip install -r requirements.txt
cd ../..
```

Moreover, install the `torch-harmonics` package for Spherical Fourier Neural Operators from the [source repository](https://github.com/NVIDIA/torch-harmonics) with the following commands

```
cd packages
git clone https://github.com/NVIDIA/torch-harmonics.git
git checkout 13aa492
cd torch-harmonics
pip install -e .
cd ../..
```

To install the CUDA versions of Deep Graph Library, follow [these instructions](https://www.dgl.ai/pages/start.html) and issue

```
pip uninstall dgl -y
pip install dgl -f https://data.dgl.ai/wheels/cu121/repo.html
pip install dglgo -f https://data.dgl.ai/wheels-test/repo.html
```

> [!IMPORTANT]
> This DGL version requires CUDA 12.1 to be installed, e.g., following [these instructions](https://developer.nvidia.com/cuda-12-1-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_local)

Finally, change into the benchmark directory, which will be considered the root directory in the following, that is, `cd src/dlwpbench`


## Navier-Stokes

To generate data and run experiments in the synthetic Navier-Stokes environment, please go to [the respective subdirectory](src/nsbench/) and follow the steps detailed there.


## WeatherBench

To download and preprocess data and run experiments in the real-world WeatherBench environment, please go to [the respective subdirectory](src/dlwpbench/) and follow the steps detailed there.

### Scoreboards

8 Prognostic Variables, 5.625° Resolution

| Model | Z500 RMSE [3/5/7/365 days] | Blow up time | Blow up T+5°C | Stable | Reference |
|-------|----------------------------|--------------|---------------|--------|-----------|
| ConvLSTM | ... | ... | ... | no | ... | 
| ConvLSTM HPX | ... | ... | ... | no | ... | 
| U-Net | ... | ... | ... | no | ... | 
| U-Net HPX | ... | ... | ... | no | ... | 
| SwinTransformer | ... | ... | ... | yes | ... | 
| SwinTransformer HPX | ... | ... | ... | yes | ... | 
| MeshGraphNet | ... | ... | ... | no | ... | 
| FNO2D | ... | ... | ... | no | ... | 
| TFNO2D | ... | ... | ... | no | ... | 
| FourCastNet p1x1 | ... | ... | ... | yes | ... | 
| FourCastNet p1x2 | ... | ... | ... | yes | ... | 
| SFNO | ... | ... | ... | yes | ... | 
| Pangu-Weather | ... | ... | ... | yes | ... | 
| GraphCast | ... | ... | ... | yes | ... | 

---
221 Prognostic Variables, 5.625° Resolution

| Model | Z500 RMSE [3/5/7/365 days] | Blow up time | Blow up T+5°C | Stable | Reference |
|-------|----------------------------|--------------|---------------|--------|-----------|

---
221 Prognostic Variables, 0.25° Resolution

| Model | Z500 RMSE [3/5/7/365 days] | Blow up time | Blow up T+5°C | Stable | Reference |
|-------|----------------------------|--------------|---------------|--------|-----------|


## Resources

Deep learning model repositories that are used in this study:

- HEALPix remapping: https://github.com/CognitiveModeling/dlwp-hpx
- Convolutional LSTM: https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py
- Fourier Neural Operator: https://github.com/neuraloperator/neuraloperator
- FourCastNet: https://github.com/NVlabs/FourCastNet
- Spherical Fourier Neural Operator: https://github.com/NVIDIA/torch-harmonics
- SwinTransformer: https://github.com/microsoft/Swin-Transformer/tree/main
- Pangu-Weather: https://github.com/lizhuoq/WeatherLearn/blob/master/weatherlearn/models/pangu/pangu.py
- MeshGraphNet: https://github.com/NVIDIA/modulus/tree/main/modulus/models/meshgraphnet
- GraphCast: https://github.com/NVIDIA/modulus/tree/main/modulus/models/graphcast

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.

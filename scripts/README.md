1. Install Apex
https://github.com/NVIDIA/apex#from-source

```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
```

2. Install pybind and python-dev
```
sudo apt install pybind11-dev python3.9-dev

export CPLUS_INCLUDE_PATH=/opt/miniconda3/include/python3.10:$CPLUS_INCLUDE_PATH
```


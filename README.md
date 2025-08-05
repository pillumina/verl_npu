# 使用方式

## 1、安装 vllm 和 vllm-ascend
```bash
# vllm
git clone -b v0.9.1 https://github.com/vllm-project/vllm.git
pip install -r vllm/requirements/build.txt -i https://mirrors.aliyun.com/pypi/simple/ # 将里面的torch==2.7删除
pip install -r vllm/requirements/common.txt -i https://mirrors.aliyun.com/pypi/simple/
cd vllm
VLLM_TARGET_DEVICE=empty python setup.py develop

# vllm-ascend
git clone -b v0.9.1-dev https://github.com/vllm-project/vllm-ascend.git
git checkout 4014ad2a46e01c79fd8d98d6283404d0bc414dce
cd vllm-ascend
pip install -v -e .
```

## 2、安装 Megatron 和 MindSpeed
```bash
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_v0.12.1
pip install -e .

git clone https://gitee.com/ascend/MindSpeed.git
git checkout ab90bbce894603b502a09025bcdf306f16b7a89f
cd MindSpeed
pip install -e .
```

## 3、安装 verl
```bash
git clone https://github.com/volcengine/verl.git
cd verl
git checkout 4e9d2878ee27e5ff004531b0a65503ed698c2c99
pip install -e .
```

## 4、安装插件
```bash
# 请确保 vllm 已正确安装并且之后不会做覆盖
git clone https://github.com/ji-huazhong/mindspeed-rl.git -b verl_npu
cd mindspeed-rl
pip install -e .
```

**注意**：安装插件前需要保证verl源码安装，否则插件不能生效。如果无法源码安装verl，需要指定verl源码路径：

```bash
VERL_PATH=path_to_verl pip install -e .
```

# SCNet 

This repository is the official implementation of [SCNet: Sparse Compression Network for Music Source Separation](https://arxiv.org/abs/2401.13276) 

![architecture](images/SCNet.png)

---
# Training
First, you need to install the requirements.

```bash
cd SCNet
pip install -r requirements.txt
```

We use the accelerate package from Hugging Face for multi-gpu training. 
```bash
accelerate config
```

You need to modify the dataset path in the /conf/config.yaml. The dataset folder should contain the train and valid parts.
```bash
data:
  wav: /path/to/dataset
```

The training command is as follows. If you do not specify a path, the default path will be used.
```bash
accelerate launch -m scnet.train --config_path path/to/config.yaml --save_path path/to/save/checkpoint/
```



---
# Citing

If you find our work useful in your research, please consider citing:
```
@misc{tong2024scnet,
      title={SCNet: Sparse Compression Network for Music Source Separation}, 
      author={Weinan Tong and Jiaxu Zhu and Jun Chen and Shiyin Kang and Tao Jiang and Yang Li and Zhiyong Wu and Helen Meng},
      year={2024},
      eprint={2401.13276},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}
```

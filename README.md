# Overview

This reposirtory contains the code used for translating about 13 million of image-caption pairs.


# Main dataset

The dataset has been downloaded from BLIP repository as:

CC3M+CC12M+SBU, Filtered synthetic caption by ViT-L, [here](https://github.com/salesforce/BLIP?tab=readme-ov-file#pre-training-datasets-download).

After downloaded the dataset, I have made chunks, just for managing the data, but it is not necessary at all.

# Setup the environment

## Create python environment to keep your system clean :)

```bash
python3 -m venv .transltion_venv
source .transltion_venv/bin/activate

```

## Install the requirements needed

pip3 install requirements.txt


# Main code used for trainslation:

The main code used for translation exists on `nllb_multi_gpus_inference file`. The code initially adopted from [here](https://github.com/FreddeFrallan/Multilingual-CLIP/tree/main/translation).

# GPU used in this translation 

- I have used a cluster with 4 A10 GPUs, each A10 GPU has 24GB of RAM.


# LICENSE

- I have taken the dataset from BLIP repo, so I kept the LICENSE of their work.

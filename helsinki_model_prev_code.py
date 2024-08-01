import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorWithPadding
from torch.utils.data import Dataset, DataLoader
from transformers import MarianMTModel, MarianTokenizer
import rich
from logger import logger
from torch.utils.data.distributed import DistributedSampler
import numpy as np

logger.info(f"Start the script")

# Importing the dataset 
# df = pd.read_feather("data/results_phase_1/ccs_synthetic.feather")

df = pd.read_csv("data/paradetox_4.csv")
logger.info(f"total dataset size is {len(df)}")

logger.info("Colmuns names")
logger.info(df.columns)
# We will translate 1/6 of the dataset

# 2092750

# Total dataset => 12,556,500

# index_min = 10000000
# index_max = 11000000

# df = df[index_min:index_max]
# df = df[:10]
# df = df[:2]
# logger.info(f"Starting translation from {index_min} to {index_max}")

# Testing the device needed 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger.info(f"The available device available is {device}")


class CaptionDataset(Dataset):
    def __init__(self, df, tokenizer_name):
        self.df = df
        self.tokenizer = MarianTokenizer.from_pretrained(tokenizer_name)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # sentence1 = df.loc[index+index_min, "caption"]


        sentence1 = df.loc[index, "neutral2"]
        # print("sentence1")
        # print(sentence1)
        if pd.isna(sentence1):
            sentence1 = "empty"
            # print("Naaaaaaaaaaaaaaaaan")

        edited_sentence1 = ">>ara<< " + sentence1

        tokens = self.tokenizer(edited_sentence1, return_tensors="pt",truncation=True)

        return tokens


model_name = "Helsinki-NLP/opus-mt-en-ar"

logger.info(f"The model used on the translation is {model_name}")

tokenizer = MarianTokenizer.from_pretrained(model_name)

logger.info(f"The max length of tokenizer is {tokenizer.model_max_length}")


model = MarianMTModel.from_pretrained(model_name, torch_dtype=torch.float16).to(device)

print(model)

logger.info(f"The model loaded on the following device: {model.device}")

model.eval()


def custom_collate_fn(data):
    """
    Data collator with padding.
    """
    tokens = [sample["input_ids"][0] for sample in data]
    attention_masks = [sample["attention_mask"][0] for sample in data]

    attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True)
    padded_tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True)

    batch = {"input_ids": padded_tokens, "attention_mask": attention_masks}
    return batch


logger.info(f"The dataset after slicing: {len(df)}")

# Loading the dataset using the CaptionDataset class 
test_data = CaptionDataset(df, "Helsinki-NLP/opus-mt-en-ar")

logger.info(f"The dataset for training length {len(test_data)}")


# Define the dataloader for the dataset 

test_dataloader = DataLoader(
    test_data,
    batch_size=256,
    shuffle=False,
    # num_workers=32,
    collate_fn=custom_collate_fn,
)

tot_test_dataloader = len(test_dataloader)

with torch.no_grad():
    decoded_tokens = []
    for i, batch in enumerate(tqdm(test_dataloader, total=tot_test_dataloader)):

        batch = {k: v.to(device) for k, v in batch.items()}
        output_tokens = model.generate(**batch)
        decoded_tokens += tokenizer.batch_decode(
            output_tokens.to("cpu"), skip_special_tokens=True
        )


df["neutral2_translated"] = decoded_tokens


df.reset_index(drop=True, inplace=True)

df.to_csv(f"data/paradetox_5.csv")


# df.to_csv(f"data/ccs_synthetic_ar_{index_min}_{index_max}.csv")
# logger.info(f"Saving csv file as ccs_synthetic_ar_{index_min}_{index_max}.csv")


# df.to_feather(f"data/ccs_synthetic_ar_{index_min}_{index_max}.feather")
# logger.info(f"Saving feather file as ccs_synthetic_ar_{index_min}_{index_max}.feather")
logger.info(f"End of the script")

#%%
from transformers import NllbMoeForConditionalGeneration, NllbTokenizer
from logger import logger
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import datetime
import json
from accelerate import Accelerator
import pandas as pd

logger.info(f"Start the script")

# Print the current time
current_time = datetime.datetime.now()
accelerator = Accelerator()

logger.info(f"Current Time: {current_time}")

# Importing the dataset 


english_caption = []
data = []

input_file = "/home/malbarham/translation_m2m_models/Parallel_Shehdeh_for_Reshma_no_duplicates.csv"
output_file = "/home/malbarham/translation_m2m_models/Parallel_Shehdeh_for_Reshma_no_duplicates_v1.csv"
model_name = "facebook/nllb-200-distilled-600M"

df = pd.read_csv(input_file)

df = df[:10]
logger.info(f"total dataset size is {len(english_caption)}")


# Testing the device needed 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]


logger.info(f"The available gpus available is {available_gpus}")


class CaptionDataset(Dataset):
    def __init__(self, df, tokenizer_name):
        self.df = df
        self.tokenizer = NllbTokenizer.from_pretrained(tokenizer_name, src_lang="eng_Latn", tgt_lang="fra_Latn")
        logger.info(f"The max length of tokenizer is {self.tokenizer.model_max_length}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        return self.df["Source"][index]
    


# Use a pipeline as a high-level helper
from transformers import pipeline

import torch 

model_name="facebook/nllb-200-distilled-600M"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = NllbTokenizer.from_pretrained(model_name, src_lang="eng_Latn", tgt_lang="arb_Arab")

from transformers import pipeline

pipe = pipeline("translation", model="facebook/nllb-200-distilled-600M",src_lang="eng_Latn", tgt_lang="arb_Arab", tokenizer=tokenizer, device_map="auto")


logger.info(f"The model used on the translation is {model_name}")


test_data = CaptionDataset(df, model_name)

logger.info(f"The dataset for training length {len(test_data)}")


# Define the dataloader for the dataset 

test_dataloader = DataLoader(
    test_data,
    batch_size=64,
    shuffle=False,
    # collate_fn=custom_collate_fn,
)

tot_test_dataloader = len(test_dataloader)

# model,training_dataloader = accelerator.prepare(model,test_dataloader)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

out_res = []
with torch.no_grad():

    for i, batch in enumerate(tqdm(test_dataloader, total=tot_test_dataloader)):

        # batch = {k: v.to(device) for k, v in batch.items()}
        # output_tokens = # model.generate(**batch)
        # decoded_tokens += tokenizer.batch_decode(output_tokens.to("cpu"), skip_special_tokens=True)
        res = pipe(batch)
        for itm in res:
            out_res.append(itm["translation_text"])


   
df["Machine_Tr"] = out_res


df.reset_index(drop=True, inplace=True)

df.to_csv(output_file)

current_time_now = datetime.datetime.now()

time_difference = current_time_now - current_time

logger.info(f"Script time {time_difference}")


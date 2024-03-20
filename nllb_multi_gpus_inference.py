import datetime
import pandas as pd
from tqdm import tqdm
from logger import logger
import torch
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM;


logger.info(f"Start the script")

# Print the current time
current_time = datetime.datetime.now()
accelerator = Accelerator()

logger.info(f"Current Time: {current_time}")

# Importing the dataset 

input_file = "/home/malbarham/Arabic-Image-Captioning-latest/ccs_synthetic_ar_8000000_10000000_translated.csv" # # Path to input file
output_file = "/home/malbarham/Arabic-Image-Captioning-latest/ccs_synthetic_ar_8000000_10000000_translated_v2.csv" # Path to output file
model_name = "facebook/nllb-200-distilled-1.3B"


# importing pandas as pd 
import pandas as pd 
  

df = pd.read_csv(input_file)

logger.info(f"Columns names: {df.columns}")

for i, itm in df.iterrows():
    if not(isinstance(itm["caption"], str)):
      logger.info(f"Row number: {i} is not string")
      logger.info(f"Row value: {itm['caption']} ")
      df.loc[i, "caption"] =  str(itm["caption"])


# Testing the device needed 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]

logger.info(f"The available gpus available is {available_gpus}")

class CaptionDataset(Dataset):
    def __init__(self, df, tokenizer_name):
        self.df = df
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, src_lang="eng_Latn")
        logger.info(f"The max length of tokenizer is {self.tokenizer.model_max_length}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        
        sentence1 = df.loc[index, "caption"]

        tokens = self.tokenizer(sentence1, return_tensors="pt", truncation=True)

        return tokens


# Use a pipeline as a high-level helper
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="eng_Latn")

model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

logger.info(f"The model used on the translation is {model_name}")

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


test_data = CaptionDataset(df, model_name)

logger.info(f"The dataset for training length {len(test_data)}")


# Define the dataloader for the dataset 

batch_size = 512

test_dataloader = DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=custom_collate_fn,
)

logger.info(f"batch_size {batch_size}")


tot_test_dataloader = len(test_dataloader)

model,training_dataloader = accelerator.prepare(model,test_dataloader)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

out_res = []

with torch.no_grad():

    for i, batch in enumerate(tqdm(test_dataloader, total=tot_test_dataloader)):

        batch = {k: v.to(device) for k, v in batch.items()}
        output_tokens = model.module.generate(**batch,  forced_bos_token_id=tokenizer.lang_code_to_id["arb_Arab"])
        out_res += tokenizer.batch_decode(output_tokens.to("cpu"), skip_special_tokens=True)

df["caption_ar_facebook_nllb-200-distilled-1.3B"] = out_res

df.reset_index(drop=True, inplace=True)

df.to_csv(output_file)

current_time_now = datetime.datetime.now()

time_difference = current_time_now - current_time

logger.info(f"Script time {time_difference}")
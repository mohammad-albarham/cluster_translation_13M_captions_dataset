#%%
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from logger import logger
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import datetime
from accelerate import Accelerator

logger.info(f"Start the script")

accelerator = Accelerator()

# Print the current time
current_time = datetime.datetime.now()

logger.info(f"Current Time: {current_time}")

# Importing the dataset 

df = pd.read_csv("translation_m2m_models/paradetox_neutral2_translated.csv")

logger.info(f"total dataset size is {len(df)}")

logger.info("Colmuns names")
logger.info(df.columns)

# index_min = 0
# index_max = 100

# df = df[index_min:index_max]

# Testing the device needed 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]


logger.info(f"The available gpus available is {available_gpus}")


class CaptionDataset(Dataset):
    def __init__(self, df, tokenizer_name):
        self.df = df
        self.tokenizer = M2M100Tokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.src_lang = "en"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # sentence1 = df.loc[index+index_min, "caption"]

        sentence1 = df.loc[index, "neutral3"]

        if pd.isna(sentence1):
            sentence1 = "empty"

        tokens = self.tokenizer(sentence1, return_tensors="pt",truncation=True)
        return tokens
    


model_name = "facebook/m2m100_1.2B"

logger.info(f"The model used on the translation is {model_name}")

# tokenizer = MarianTokenizer.from_pretrained(model_name)
tokenizer = M2M100Tokenizer.from_pretrained(model_name)


logger.info(f"The max length of tokenizer is {tokenizer.model_max_length}")


model = M2M100ForConditionalGeneration.from_pretrained(model_name).to(device)

# model = MarianMTModel.from_pretrained(model_name, torch_dtype=torch.float16).to(device)


# print(model)

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


# logger.info(f"The dataset after slicing: {len(df)}")

# Loading the dataset using the CaptionDataset class 
test_data = CaptionDataset(df, model_name)

logger.info(f"The dataset for training length {len(test_data)}")


# Define the dataloader for the dataset 

test_dataloader = DataLoader(
    test_data,
    batch_size=8,
    shuffle=False,
    # num_workers=32,
    collate_fn=custom_collate_fn,
)

tot_test_dataloader = len(test_dataloader)

model,training_dataloader = accelerator.prepare(model,test_dataloader)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


with torch.no_grad():
    decoded_tokens = []
    for i, batch in enumerate(tqdm(test_dataloader, total=tot_test_dataloader)):

        batch = {k: v.to(device) for k, v in batch.items()}
        # output_tokens = model.generate(**batch)
        output_tokens = model.module.generate(**batch, forced_bos_token_id=tokenizer.get_lang_id("ar"))
        # decoded_tokens += tokenizer.batch_decode(
        #     output_tokens.to("cpu"), skip_special_tokens=True
        # )
        decoded_tokens += tokenizer.batch_decode(output_tokens.to("cpu"), skip_special_tokens=True)



df["neutral3_translated"] = decoded_tokens


df.reset_index(drop=True, inplace=True)

df.to_csv(f"/home/malbarham/translation_m2m_models/paradetox_neutral3_translated.csv")


current_time_now = datetime.datetime.now()

time_difference = current_time_now - current_time

logger.info(f"End of the script {time_difference}")


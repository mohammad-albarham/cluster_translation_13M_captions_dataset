import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorWithPadding
from torch.utils.data import Dataset, DataLoader
from transformers import MarianMTModel, MarianTokenizer
from accelerate import Accelerator

print("Start the code ...")

accelerator = Accelerator()

print("Start reading the data using pandas ...")

index_min = 0
index_max = 10000000
# Importing the dataset 
df = pd.read_feather("data/laion_synthetic_filtered_large_20000000.feather")  # laion_synthetic_filtered_large_10000000.feather
print("Before slicing 1: ", len(df))

df = df[index_min:index_max]
print("After slicing 1: ", len(df))

print("Finish reading the data using pandas ...")

class CaptionDataset(Dataset):
    def __init__(self, df, tokenizer_name):
        self.df = df
        self.tokenizer = MarianTokenizer.from_pretrained(tokenizer_name)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        sentence1 = df.loc[index+index_min, "caption"]

        edited_sentence1 = ">>ara<< " + sentence1

        tokens = self.tokenizer(edited_sentence1, return_tensors="pt")

        return tokens


print("Start loading the model ...")

model_name = "Helsinki-NLP/opus-mt-en-ar"

tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

print("End loading the model ...")

print(model)

model.eval()

if torch.cuda.device_count() > 1:
    print(torch.cuda.device_count())

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


print("After slicing: ", len(df))

# Loading the dataset using the CaptionDataset class 
test_data = CaptionDataset(df, "Helsinki-NLP/opus-mt-en-ar")


# Define the dataloader for the dataset 

test_dataloader = DataLoader(
    test_data,
    batch_size=64,
    shuffle=False,
    collate_fn=custom_collate_fn,
)

tot_test_dataloader = len(test_dataloader)



model,training_dataloader = accelerator.prepare(model,test_dataloader)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


with torch.no_grad():
    decoded_tokens = []
    for i, batch in enumerate(tqdm(test_dataloader, total=tot_test_dataloader)):

        try:

            batch = {k: v.to(device) for k, v in batch.items()}
            output_tokens = model.module.generate(**batch)
            decoded_tokens += tokenizer.batch_decode(
                output_tokens.to("cpu"), skip_special_tokens=True
            )
            # print(decoded_tokens)
        except:
            decoded_tokens += ["out of memory"] *64
            print("out of memory: ", i)


df["caption_ar"] = decoded_tokens


df.reset_index(drop=True, inplace=True)

df.to_feather(f"data/ccs_synthetic_ar_{index_min}_{index_max}_20M.feather")

df.to_csv(f"data/ccs_synthetic_ar_{index_min}_{index_max}_20M.csv")


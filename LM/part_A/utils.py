import torch
import torch.utils.data as data
from transformers import AutoTokenizer
from functools import partial
from torch.utils.data import DataLoader

def read_file(path, eos_token="<eos>"):
    output = []
    with open(path, "r") as f:
        for line in f.readlines():
            output.append(line.strip() + " " + eos_token)

    return output

class PennTreeBank (data.Dataset):
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, corpus):
        self.sents = [sent for sent in corpus]

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        return self.sents[idx]
    
def collate_fn(batch, tokenizer, device):
    tokenized = tokenizer(batch, padding=True, return_tensors="pt")

    input_ids = tokenized.input_ids[:, :-1].detach().clone().to(device)
    # labels are the input shifted left -> predict the next token
    labels = tokenized.input_ids[:, 1:].detach().clone().to(device)

    # count non-pad tokens
    n_tokens = torch.sum(input_ids != tokenizer.pad_token_id)

    return input_ids, labels, n_tokens

def instantiate_loader(train_path="dataset/PennTreeBank/ptb.train.txt", dev_path="dataset/PennTreeBank/ptb.valid.txt", test_path="dataset/PennTreeBank/ptb.test.txt", dev='cpu', trainb_size=8, devb_size=16, testb_size=16):
    train_raw = read_file(train_path)
    dev_raw = read_file(dev_path)
    test_raw = read_file(test_path)

    train_dataset = PennTreeBank(train_raw)
    dev_dataset = PennTreeBank(dev_raw)
    test_dataset = PennTreeBank(test_raw)

    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    train_loader = DataLoader(train_dataset, batch_size=trainb_size, collate_fn=partial(collate_fn, tokenizer=tokenizer, device=dev),  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=devb_size, collate_fn=partial(collate_fn, tokenizer=tokenizer, device=dev))
    test_loader = DataLoader(test_dataset, batch_size=testb_size, collate_fn=partial(collate_fn, tokenizer=tokenizer, device=dev))

    return train_loader, dev_loader, test_loader, tokenizer


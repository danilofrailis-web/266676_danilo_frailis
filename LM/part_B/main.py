# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

from functions import *
from utils import *
from model import *
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.optim as optim

DEVICE = "cuda:0"


if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    train_loader, dev_loader, test_loader, tokenizer = instantiate_loader(dev=DEVICE)

    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_len = len(tokenizer)

    alphas = [16]
    for alpha in alphas:
        model = GPT2_LoRA.from_pretrained("openai-community/gpt2", alpha=alpha, rank=alpha)
        model.to(DEVICE)

        lr = 0.0003

        #train only adapter layers
        for param in model.parameters():
            #freeze all parameters
            param.requires_grad = False
        for module in model.modules():
            #make only your layers trainable
            if hasattr(module,  "lora_layers"):
                #change the fucking layer name
                for param in module.lora_layers.parameters():
                    param.requires_grad = True

        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)

        optimizer = optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr = lr
        )

        param_stats(model)

        current_model, ppl, loss = run(model=model,
                                    train_loader=train_loader,
                                    dev_loader=dev_loader,
                                    test_loader=test_loader,
                                    optimizer=optimizer,
                                    tokenizer=tokenizer,
                                    n_epochs=6,
                                    patience=3,
                                    device=DEVICE)

        path = f'bin/lr0003_lora_a{alpha}_ra.pt'
        torch.save(current_model.state_dict(), path)
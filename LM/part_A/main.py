# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
from utils import *
from model import *

import torch.optim as optim

DEVICE = 'cuda:0'


if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    train_loader, dev_loader, test_loader, tokenizer = instantiate_loader(dev=DEVICE)

    vocab_len = len(tokenizer)
    lr = 0.0003
    d_model=64
    n_heads = 4
    n_layers = 4
    ff_dim=256

    
    criterion_train = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    criterion_eval = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)



    model = Model(
        vocab_len,
        pos_embed_size=1024,
        d_model=20,
        n_heads=1,
        num_layers=1,
        ff_dim=20
    ).to(DEVICE)
    model.apply(init_weights)


    optimizer = optim.AdamW(model.parameters(), lr=lr)

    current_model, ppl, loss = run(model=model,
                        train_loader= train_loader,
                        dev_loader=dev_loader,
                        test_loader=test_loader,
                        criterion_train=criterion_train,
                        criterion_eval=criterion_eval,
                        optimizer=optimizer,
                        n_epochs=15,
                        patience=3,
                        device=DEVICE)

    print(ppl)

    path = f'bin/baseline.pt'
    torch.save(current_model.state_dict(), path)


    model = Model(
        vocab_len,
        pos_embed_size=1024,
        d_model=128,
        n_heads=1,
        num_layers=1,
        ff_dim=20
    ).to(DEVICE)
    model.apply(init_weights)


    optimizer = optim.AdamW(model.parameters(), lr=lr)

    current_model, ppl, loss = run(model=model,
                        train_loader= train_loader,
                        dev_loader=dev_loader,
                        test_loader=test_loader,
                        criterion_train=criterion_train,
                        criterion_eval=criterion_eval,
                        optimizer=optimizer,
                        n_epochs=15,
                        patience=3,
                        device=DEVICE)

    print(ppl)

    path = f'bin/lr0003_dmodel28.pt'
    torch.save(current_model.state_dict(), path)
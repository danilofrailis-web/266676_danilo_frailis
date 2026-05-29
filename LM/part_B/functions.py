import math
from tqdm.auto import tqdm
import torch
import copy

def param_stats(model):
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"total params: {total:,}")
    print(f"trainable params: {trainable:,}")
    print(f"frozen params: {total - trainable:,}")

def train_loop(data, optimizer, model, tokenizer):
    model.train()
    loss_array = []
    number_of_tokens = []

    pbar = tqdm(data, desc="Training:", unit="batch", total=len(data))

    for i, (input_ids, _, n_tokens) in enumerate(pbar):
        optimizer.zero_grad()
        labels = input_ids.clone().detach()
        labels[labels==tokenizer.pad_token_id] = -100
        output = model(input_ids, labels=labels)
        loss_array.append(output.loss.item()*n_tokens)
        number_of_tokens.append(n_tokens)
        output.loss.backward() #computes the gradient deleting the computationale graph
        optimizer.step()

        if i%100 == 0:
            pbar.set_postfix(loss=(sum(loss_array)/sum(number_of_tokens)).item())

    return sum(loss_array)/sum(number_of_tokens)

def eval_loop(data, model, tokenizer):
    model.eval()
    loss_to_return = []
    loss_array = []
    number_of_tokens = []
    with torch.no_grad():
        for input_ids, _, n_tokens in tqdm(data, desc="Evaluating: ", unit="batch", total=len(data)):
            labels = input_ids.clone().detach()
            labels[labels==tokenizer.pad_token_id] = -100
            output = model(input_ids, labels=labels)
            loss_array.append(output.loss.item()*n_tokens)
            number_of_tokens.append(n_tokens)

    loss_to_return = sum(loss_array)/sum(number_of_tokens)
    ppl = math.exp(loss_to_return)
    return ppl, loss_to_return

def run(model, train_loader, dev_loader, test_loader, optimizer, n_epochs, tokenizer, patience, device='cpu'):
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    pat = patience
    best_ppl = math.inf
    best_model = None
    pbar = tqdm(range(n_epochs))

    for epoch in pbar:
        loss = train_loop(train_loader, optimizer, model, tokenizer)
        if epoch%1 == 0:
            sampled_epochs.append(epoch)
            losses_train.append(loss.item())
            ppl_dev, loss_dev = eval_loop(dev_loader, model, tokenizer)
            losses_dev.append(loss_dev.item())
            pbar.set_description("PPL: %f" %ppl_dev)

            if ppl_dev < best_ppl:
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to(device)
                pat = patience
            else:
                pat -= 1

            if pat <= 0:
                break

    best_model.to(device)
    final_ppl, final_loss = eval_loop(test_loader, best_model, tokenizer)
    print(f'Test ppl: {final_ppl}\nTest loss: {final_loss}')
    return best_model, final_ppl, final_loss
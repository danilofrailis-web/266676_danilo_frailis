import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import copy

from tqdm.auto import tqdm


def train_loop(data, optimizer, criterion, model):

    model.train()
    loss_array = []
    number_of_tokens = []

    pbar = tqdm(data, desc="Training:", unit="batch", total=len(data))

    for i, (input_ids, labels, n_tokens) in enumerate(pbar):
        optimizer.zero_grad()
        output = model(input_ids)

        #reshape to (B, vocab, L)
        loss = criterion(output.permute(0, 2, 1), labels)
        loss_array.append(loss.item() * n_tokens)
        number_of_tokens.append(n_tokens)
        
        #compute the gradient and delete the computational graph
        loss.backward()
        optimizer.step()

        if i%100 == 0:
            pbar.set_postfix(loss=(sum(loss_array)/sum(number_of_tokens)).item())

    return sum(loss_array)/sum(number_of_tokens)

def eval_loop(data, eval_criterion, model):
    model.eval()
    loss_to_return = []
    loss_array = []
    number_of_tokens = []

    # softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability

    with torch.no_grad():
        for input_ids, labels, n_tokens in tqdm(data, desc="Evaluating: ", unit="batch", total=len(data)):
            output = model(input_ids)

            #reshape into (B, vocab, L)
            loss = eval_criterion(output.permute(0, 2, 1), labels)
            loss_array.append(loss.item() * n_tokens)
            number_of_tokens.append(n_tokens)
    
    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    ppl = math.exp(loss_to_return)
    return ppl, loss_to_return

def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.Linear]:
            torch.nn.init.uniform_(m.weight, -0.01, 0.01)
            if m.bias != None:
                m.bias.data.fill_(0.01)

def run(model, train_loader, dev_loader, test_loader, criterion_train, criterion_eval, optimizer, n_epochs, patience, device='cpu'):
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None
    pbar = tqdm(range(n_epochs))
    pat = patience

    for epoch in pbar:
        loss = train_loop(train_loader, optimizer, criterion_train, model)

        if epoch%1 == 0:
            sampled_epochs.append(epoch)
            losses_train.append(loss.item())
            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
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
    final_ppl, final_loss = eval_loop(test_loader, criterion_eval, best_model)
    print(f'Test ppl: {final_ppl}\nTest loss: {final_loss}')
    return best_model, final_ppl, final_loss
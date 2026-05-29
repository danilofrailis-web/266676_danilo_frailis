import torch
import torch.nn as nn
from tqdm.auto import tqdm
from conll import evaluate
from sklearn.metrics import classification_report 
import numpy as np
import matplotlib.pyplot as plt
import copy



def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.Linear]:
            torch.nn.init.uniform_(m.weight, -0.01, 0.01)
            if m.bias != None:
                m.bias.data.fill_(0.01)

def train_loop(data, optimizer, criterion_slots, criterion_intents, model):
    model.train()
    loss_array = []

    for i, batch in enumerate(data):
        optimizer.zero_grad() # Zeroing the gradient

        slots, intent = model(batch['utterances'], batch['slots_len'])
        slots = slots.permute(0,2,1) # We need this for computing the loss

        loss_intent = criterion_intents(intent, batch['intents'])
        loss_slot = criterion_slots(slots, batch['y_slots'])
        loss = loss_intent + loss_slot # In joint training we sum the losses. 
                                       # Is there another way to do that?
        loss_array.append(loss.item())
        loss.backward() # Compute the gradient, deleting the computational graph
        optimizer.step() # Update the weights

    return loss_array

def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    model.eval()
    loss_array = []
    
    ref_intents = []
    hyp_intents = []
    
    ref_slots = []
    hyp_slots = []
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for batch in data:
            slots, intents = model(batch['utterances'], batch['slots_len'])
            slots = slots.permute(0,2,1) # We need this for computing the loss
            loss_intent = criterion_intents(intents, batch['intents'])
            loss_slot = criterion_slots(slots, batch['y_slots'])
            loss = loss_intent + loss_slot 
            loss_array.append(loss.item())

            # Intent inference
            # Get the most probable class
            out_intents = [lang.id2intent[x] for x in torch.argmax(intents, dim=1).tolist()] 
            gt_intents = [lang.id2intent[x] for x in batch['intents'].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)
            
            # Slot inference 
            output_slots = torch.argmax(slots, dim=1)
            for id_seq, seq in enumerate(output_slots):
                length = batch['slots_len'].tolist()[id_seq] - 1 # -1, we ignore the CLS

                utt_ids = batch['utterances'][id_seq][:length].tolist()
                gt_ids = batch['y_slots'][id_seq][:length].tolist()
                gt_slots = [lang.id2slot[elem] for elem in gt_ids]
                utterance = [lang.id2word[elem] for elem in utt_ids]

                to_decode = seq[:length].tolist()
                ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
                hyp_slots.append(tmp_seq)
    try:            
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        # Sometimes the model predicts a class that is not in REF
        print("Warning:", ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s))
        results = {"total":{"f":0}}
        
    report_intent = classification_report(ref_intents, hyp_intents, 
                                          zero_division=False, output_dict=True)
    return results, report_intent, loss_array

def run(train_loader, dev_loader, test_loader, optimizer, criterion_slots, criterion_intents, model, lang, n_epochs=200, patience=2):
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_f1 = 0
    pat = patience
    best_model = None


    pbar = tqdm(range(n_epochs))
    for x in pbar:
        loss = train_loop(train_loader, optimizer, criterion_slots, 
                          criterion_intents, model)
        if x % 1 == 0:
            sampled_epochs.append(x)
            losses_train.append(np.asarray(loss).mean())
            results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots, 
                                                          criterion_intents, model, lang)

            pbar.set_description(f"Slot F1: {results_dev['total']['f']:.2f}; Intent Acc: {intent_res['abbreviation']['f1-score']:.2f}")
            losses_dev.append(np.asarray(loss_dev).mean())

            f1 = results_dev['total']['f']

            if f1 > best_f1:
                best_f1 = f1
                best_model = copy.deepcopy(model.state_dict())
                pat = patience
            else:
                pat -= 1
            if pat <= 0: # Early stopping with patient
                break # Not nice but it keeps the code clean
    model.load_state_dict(best_model)
    results_test, intent_test, loss_array_test = eval_loop(test_loader, criterion_slots, 
                                             criterion_intents, model, lang)

    return model, results_test, intent_test, loss_array_test


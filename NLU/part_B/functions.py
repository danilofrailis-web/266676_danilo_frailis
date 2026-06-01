import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from conll import evaluate
import numpy as np
from tqdm.auto import tqdm
import copy
from sklearn.metrics import accuracy_score


def train_loop(data, optimizer, criterion_slots, criterion_intents, model):
    model.train()
    loss_array = []
    print(len(data))
    for batch in data:
        optimizer.zero_grad()
        # HF models require input_ids and attention_mask
        slots, intent = model(batch['input_ids'], batch['attention_mask'])
        
        # Reshape for CrossEntropy (Batch, Classes, SeqLen)
        slots = slots.permute(0, 2, 1) 

        loss_intent = criterion_intents(intent, batch['intent'])
        # CrossEntropy ignores -100 (sub-tokens/padding) automatically
        loss_slot = criterion_slots(slots, batch['slots'])
        
        loss = loss_intent + loss_slot
        loss.backward()
        optimizer.step()
        loss_array.append(loss.item())

    return loss_array

def eval_loop(data, model, lang):
    model.eval()
    ref_intents, hyp_intents = [], []
    ref_slots, hyp_slots = [], []
    
    with torch.no_grad():
        for batch in data:
            slot_logits, intent_logits = model(batch['input_ids'], batch['attention_mask'])
            
            # 1. Intent Logic
            hyp_intents.extend(torch.argmax(intent_logits, dim=-1).cpu().numpy())
            ref_intents.extend(batch['intent'].cpu().numpy())
            
            # 2. Slot Logic (Handle Sub-tokens for conll.py)
            preds = torch.argmax(slot_logits, dim=-1).cpu().numpy()
            targets = batch['slots'].cpu().numpy()
            
            for i in range(len(preds)):
                # Only keep predictions where target is not -100
                mask = targets[i] != -100 
                word_preds = preds[i][mask]
                word_targets = targets[i][mask]
                
                # conll.evaluate expects list of (None, Label) tuples for alignment
                ref_slots.append([(None, lang.id2slot[t]) for t in word_targets])
                hyp_slots.append([(None, lang.id2slot[p]) for p in word_preds])
                
    # Calculate Metrics
    intent_report = classification_report(
        ref_intents, 
        hyp_intents, 
        labels=list(lang.intent2id.values()),   # Pass all possible label IDs
        target_names=list(lang.intent2id.keys()), # Pass all corresponding names
        output_dict=True, 
        zero_division=0
    )
    print(intent_report.keys())
    intent_report["accuracy"] = accuracy_score(ref_intents, hyp_intents)
    slot_results = evaluate(ref_slots, hyp_slots)
    
    return slot_results, intent_report

def run(train_loader, dev_loader, test_loader, model, optimizer, criterion_slots, criterion_intents, lang, n_epochs=50, patience=5):
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    
    best_f1 = 0
    pat = patience
    best_model = None

    pbar = tqdm(range(n_epochs))
    for epoch in pbar:
        train_loss = train_loop(train_loader, optimizer, criterion_slots, criterion_intents, model)
        
        sampled_epochs.append(epoch)
        losses_train.append(np.asarray(train_loss).mean())
        
        slot_results, intent_report = eval_loop(dev_loader, model, lang)
        f1  = slot_results['total']['f']
        acc = intent_report['accuracy']

        pbar.set_description(f"Slot F1: {f1:.2f}; Intent Acc: {acc:.2f}")

        if f1 > best_f1:
            best_f1 = f1
            best_model = copy.deepcopy(model).state_dict()
            pat = patience
        else:
            pat -= 1

        if pat <= 0:
            break

    model.load_state_dict(best_model)
    results_test, intent_test = eval_loop(test_loader, model, lang)
    return results_test, intent_test, best_model


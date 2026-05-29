# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
from utils import *
from model import *
import os
import torch.optim as optim


DEVICE = 'cpu'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    train_raw, dev_raw, test_raw = load_datasets()

    w2id, slot2id, intent2id = create_dictionaries(train_raw, dev_raw, test_raw)
    

    # No set() since we want to compute the cutoff
    words = sum([x['utterance'].split() for x in train_raw], []) # sum(list[list], []) -> from list of list to list
    # We do not want unk labels (slots), 
    # however this depends on the research purpose
    corpus = train_raw + dev_raw + test_raw 
    slots = set(sum([line['slots'].split() for line in corpus],[]))
    intents = set([line['intent'] for line in corpus])

    # words are only from te training set
    # labels from the whole corpus (we do not want unk labels)
    lang = Lang(words, intents, slots, cutoff=0)

    # Create our datasets
    train_dataset = IntentsAndSlots(train_raw, lang)
    dev_dataset = IntentsAndSlots(dev_raw, lang)
    test_dataset = IntentsAndSlots(test_raw, lang)

    #dataloader instantiations
    train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn,  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)

    learning_rates = [0.001, 0.0005, 0.0003, 0.0001]



    n_epochs = 200
    runs = 5
    vocab_len = len(lang.word2id)
    slots_len = len(lang.id2slot) # pad & cls have the same id
    n_intents = len(lang.intent2id)

    for lr in learning_rates:

        slot_f1s, intent_acc = [], []

        for x in tqdm(range(0, runs)):
            model = GPT2(
                vocab_len,
                slots_len,
                n_intents,
                pos_emb_size=1024,
                d_model=20,
                n_heads=1,
                num_layers=1,
                ff_dim=20,
            ).to(DEVICE)
            model.apply(init_weights)

            optimizer = optim.AdamW(model.parameters(), lr=lr)
            criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
            criterion_intents = nn.CrossEntropyLoss()

            results_test, intent_test, loss_array_test = run(train_loader=train_loader, 
                                                            dev_loader=dev_loader,
                                                            test_loader=test_loader,
                                                            optimizer=optimizer,
                                                            criterion_slots=criterion_slots,
                                                            criterion_intents=criterion_intents,
                                                            model=model,
                                                            lang=lang,
                                                            n_epochs=n_epochs)
            
            intent_acc.append(intent_test['accuracy'])
            slot_f1s.append(results_test['total']['f'])
        
        slot_f1s = np.asarray(slot_f1s)
        intent_acc = np.asarray(intent_acc)
        print('Slot F1', round(slot_f1s.mean(),3), '+-', round(slot_f1s.std(),3))
        print('Intent Acc', round(intent_acc.mean(), 3), '+-', round(slot_f1s.std(), 3))
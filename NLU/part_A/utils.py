import json
from pprint import pprint
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import os
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader

# Add functions or classes used for data loading and preprocessing

PAD_TOKEN = 0
DEVICE = 'cuda:0'


def load_data(path):
    dataset = []

    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset


def load_datasets():
    portion = 0.10

    tmp_train_raw = load_data(os.path.join('dataset','ATIS','train.json'))
    test_raw = load_data(os.path.join('dataset','ATIS', 'test.json'))
    print('Train samples:', len(tmp_train_raw))
    print('Test samples:', len(test_raw))
    
    intents = [x['intent'] for x in tmp_train_raw] #stratification on intents
    count_y = Counter(intents)

    labels = []
    inputs = []
    mini_train = []

    for id_y, y in enumerate(intents):
        if count_y[y] > 1:# If some intents occurs only once, we put them in training
            inputs.append(tmp_train_raw[id_y])
            labels.append(y)
        else:
            mini_train.append(tmp_train_raw[id_y])

    #random stratify
    X_train, X_dev, Y_train, Y_dev = train_test_split(inputs, 
                                                      labels,
                                                      test_size=portion,
                                                      random_state=42,
                                                      shuffle=True,
                                                      stratify=labels)

    X_train.extend(mini_train)
    train_raw = X_train
    dev_raw = X_dev

    Y_test = [x['intent'] for x in test_raw] 

    print('Train:')
    pprint({k:round(v/len(Y_train),3)*100 for k, v in sorted(Counter(Y_train).items())})
    print('Dev:'), 
    pprint({k:round(v/len(Y_dev),3)*100 for k, v in sorted(Counter(Y_dev).items())})
    print('Test:') 
    pprint({k:round(v/len(Y_test),3)*100 for k, v in sorted(Counter(Y_test).items())})
    print('='*89)
    # Dataset size
    print('TRAIN size:', len(train_raw))
    print('DEV size:', len(dev_raw))
    print('TEST size:', len(test_raw))

    return train_raw, dev_raw, test_raw

class Lang():
    def __init__(self, words, intents, slots, cutoff=0, cls=True):
        self.word2id = self.w2id(words, cutoff=cutoff, unk=True, cls=cls)
        self.slot2id = self.lab2id(slots, cls=cls)
        self.intent2id = self.lab2id(intents, pad=False, cls=False)
        self.id2word = {v:k for k, v in self.word2id.items()}
        # cls will have the same id as the pad token
        self.id2slot = {v:k for k, v in self.slot2id.items() if not cls or k != 'cls'}
        self.id2intent = {v:k for k, v in self.intent2id.items()}
        
    def w2id(self, elements, cutoff=None, unk=True, cls=True):
        vocab = {'pad': PAD_TOKEN}
        if unk:
            vocab['unk'] = len(vocab)
        if cls:
            vocab['cls'] = len(vocab)
        count = Counter(elements)
        for k, v in count.items():
            if v > cutoff:
                vocab[k] = len(vocab)
        return vocab
    
    def lab2id(self, elements, pad=True, cls=True):
        vocab = {}
        if pad:
            vocab['pad'] = PAD_TOKEN
        for elem in elements:
            vocab[elem] = len(vocab)
        if cls:
            # when predicting the slots, we want to ignore the CLS
            # CLS will only be used for intent classification
            vocab['cls'] = PAD_TOKEN
        return vocab
    
class IntentsAndSlots(data.Dataset):
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, dataset, lang, unk='unk', cls='cls', add_cls=True):
        self.utterances = []
        self.intents = []
        self.slots = []
        self.unk = unk
        self.cls = cls
        self.add_cls = add_cls
        
        for x in dataset:
            self.utterances.append(x['utterance'])
            self.slots.append(x['slots'])
            self.intents.append(x['intent'])

        self.utt_ids = self.mapping_seq(self.utterances, lang.word2id)
        self.slot_ids = self.mapping_seq(self.slots, lang.slot2id)
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        utt = torch.Tensor(self.utt_ids[idx])
        slots = torch.Tensor(self.slot_ids[idx])
        intent = self.intent_ids[idx]
        sample = {'utterance': utt, 'slots': slots, 'intent': intent}
        return sample
    
    # Auxiliary methods
    
    def mapping_lab(self, data, mapper):
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]
    
    def mapping_seq(self, data, mapper): # Map sequences to number
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq.split():
                if x in mapper:
                    tmp_seq.append(mapper[x])
                else:
                    tmp_seq.append(mapper[self.unk])
            if self.add_cls:
                tmp_seq.append(mapper[self.cls])
            res.append(tmp_seq)
        return res
    
def collate_fn(data):
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len 
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape 
        # batch_size * maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(PAD_TOKEN)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq # We copy each sequence into the matrix
        return padded_seqs, lengths

    data_by_key = {}
    for key in data[0].keys():
        data_by_key[key] = [d[key] for d in data]
        
    # We just need one length for packed pad seq, since len(utt) == len(slots)
    src_utt, _ = merge(data_by_key['utterance'])
    y_slots, y_lengths = merge(data_by_key["slots"])
    intent = torch.LongTensor(data_by_key["intent"])
    
    src_utt = src_utt.to(DEVICE) # We load the Tensor on our selected device
    y_slots = y_slots.to(DEVICE)
    intent = intent.to(DEVICE)
    y_lengths = torch.LongTensor(y_lengths).to(DEVICE)
    
    new_item = {}
    new_item["utterances"] = src_utt
    new_item["intents"] = intent
    new_item["y_slots"] = y_slots
    new_item["slots_len"] = y_lengths
    return new_item


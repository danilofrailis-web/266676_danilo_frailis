import json
from sklearn.model_selection import train_test_split
from collections import Counter
import os
import torch
import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence


# Add functions or classes used for data loading and preprocessing

PAD_TOKEN = 0


def load_data(path):
    dataset = []

    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset


def load_datasets():
    portion = 0.10

    tmp_train_raw = load_data(os.path.join('dataset','ATIS','train.json'))
    test_raw = load_data(os.path.join('dataset','ATIS', 'test.json'))
    
    portion = 0.10
    intents = [x['intent'] for x in tmp_train_raw] # We stratify on intents
    count_y = Counter(intents)

    labels = []
    inputs = []
    mini_train = []

    for id_y, y in enumerate(intents):
        if count_y[y] > 1: # If some intents occurs only once, we put them in training
            inputs.append(tmp_train_raw[id_y])
            labels.append(y)
        else:
            mini_train.append(tmp_train_raw[id_y])
    # Random Stratify
    X_train, X_dev, y_train, y_dev = train_test_split(inputs, labels, test_size=portion, 
                                                        random_state=42, 
                                                        shuffle=True,
                                                        stratify=labels)
    X_train.extend(mini_train)
    train_raw = X_train
    dev_raw = X_dev

    return train_raw, dev_raw, test_raw

class Lang():
    def __init__(self, words, intents, slots, cutoff=0, cls=True):
        self.word2id = self.w2id(words, cutoff=cutoff, unk=True, cls=cls)
        self.slot2id = self.lab2id(slots, cls=cls)
        self.intent2id = self.lab2id(intents, pad=False, cls=False)
        self.id2word = {v:k for k, v in self.word2id.items()}
        # cls will have the same id as the pad token
        self.id2slot = {v:k for k, v in self.slot2id.items() if k != 'cls'}
        self.id2intent = {v:k for k, v in self.intent2id.items()}
        
    def w2id(self, elements, cutoff=0, unk=True, cls=True):
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
    def __init__(self, dataset, tokenizer, slot2id, intent2id):
        self.tokenizer = tokenizer
        self.utterances = []
        self.slots = []
        self.intents = []

        for item in dataset:
            words = item['utterance'].split()
            slots = item['slots'].split()
            
            #tokenization and alignment of labels
            encoding = tokenizer(words, is_split_into_words=True, truncation=True)
            #mapping of tokens to original word indices
            word_ids = encoding.word_ids()

            label_ids = []
            previous_word_idx = None

            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    #first subtoken of the word
                    label_ids.append(slot2id[slots[word_idx]])
                else:
                    #other subtokens of the same word are ignored
                    label_ids.append(-100)
                previous_word_idx = word_idx
            
            self.utterances.append(torch.tensor(encoding['input_ids']))
            self.slots.append(torch.tensor(label_ids))
            self.intents.append(intent2id[item['intent']])

    def __len__(self):
        return len(self.utterances)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.utterances[idx],
            'slots': self.slots[idx],
            'intent': self.intents[idx]
        }  

def collate_fn(data, device):
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
    
    src_utt = src_utt.to(device) # We load the Tensor on our selected device
    y_slots = y_slots.to(device)
    intent = intent.to(device)
    y_lengths = torch.LongTensor(y_lengths).to(device)
    
    new_item = {}
    new_item["utterances"] = src_utt
    new_item["intents"] = intent
    new_item["y_slots"] = y_slots
    new_item["slots_len"] = y_lengths
    return new_item

def collate_hf_huggingface(batch, device, pad_token_id):
    input_ids = [item['input_ids'] for item in batch]
    slots = [item['slots'] for item in batch]
    intents = torch.tensor([item['intent'] for item in batch]).to(device)

    padded_input = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id).to(device)
    padded_slots = pad_sequence(slots, batch_first=True, padding_value=-100).to(device)

    attention_mask = (padded_input != pad_token_id).float().to(device)

    return {
        'input_ids': padded_input,
        'attention_mask': attention_mask,
        'slots': padded_slots,
        'intent': intents
    }

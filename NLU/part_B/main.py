from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from functools import partial
from utils import *
from functions import *
from model import *
import warnings

DEVICE = 'cuda:0'
MODEL_NAME = 'openai-community/gpt2' # 'bert-base-uncased'

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)

    train_raw, dev_raw, test_raw = load_datasets()
    print(f"Data Loaded: {len(train_raw)} training examples")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True)

    #GPT2 needs a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    corpus = train_raw + dev_raw + test_raw
    slots = set(sum([x['slots'].split() for x in corpus], []))
    intents = set([x['intent'] for x in corpus])
    lang = Lang([], intents, slots, cls=False)

    #datasets & loaders
    collate_fn = partial(collate_hf_huggingface, device=DEVICE, pad_token_id=tokenizer.pad_token_id)

    train_loader = DataLoader(IntentsAndSlots(train_raw, tokenizer, lang.slot2id, lang.intent2id), 
                              batch_size=16, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(IntentsAndSlots(dev_raw, tokenizer, lang.slot2id, lang.intent2id), 
                              batch_size=16, collate_fn=collate_fn)
    test_loader = DataLoader(IntentsAndSlots(test_raw, tokenizer, lang.slot2id, lang.intent2id), 
                              batch_size=16, shuffle=True, collate_fn=collate_fn)

    #model
    model = HuggingFaceModel(MODEL_NAME, len(lang.slot2id), len(lang.intent2id)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005)
    criterion_slots = nn.CrossEntropyLoss(ignore_index=-100)
    criterion_intents = nn.CrossEntropyLoss()

    results_test, intent_test, best_model = run(train_loader, dev_loader, test_loader,
                                       model, optimizer, criterion_slots, criterion_intents,
                                       lang, n_epochs=20, patience=3)
    
    model.load_state_dict(best_model)

    PATH = os.path.join("bin", f"GPT2_lr0.00005.pt")
    saving_object = {
            "model": model, 
            "optimizer": optimizer.state_dict(), 
            "w2id": lang.word2id, 
            "slot2id": lang.slot2id, 
        "intent2id": lang.intent2id}
    torch.save(saving_object, PATH)
    print(f"\nFinal Test Slot F1: {results_test['total']['f']:.4f}")
    print(f"Final Test Intent Acc: {intent_test['accuracy']:.4f}")


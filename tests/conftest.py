import pytest,torch
from tokenizers import Tokenizer,models,pre_tokenizers,processors
from transformers import PreTrainedTokenizerFast

@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(11);torch.set_num_threads(2)

@pytest.fixture
def tokenizer():
    words=['[PAD]','[UNK]','[BOS]','[EOS]','a','cat','dog','on','desk','red','blue','with','hat',
           'micro','##scope','mouse','computer','and','small','wooden','table','.','the','in','garden']
    vocab={w:i for i,w in enumerate(words)}
    t=Tokenizer(models.WordPiece(vocab,unk_token='[UNK]'))
    t.pre_tokenizer=pre_tokenizers.Whitespace()
    t.post_processor=processors.TemplateProcessing(single='[BOS] $A [EOS]',special_tokens=[('[BOS]',2),('[EOS]',3)])
    return PreTrainedTokenizerFast(tokenizer_object=t,pad_token='[PAD]',unk_token='[UNK]',bos_token='[BOS]',eos_token='[EOS]',model_max_length=24)

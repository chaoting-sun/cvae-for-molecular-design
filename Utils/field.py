import os
import re
import pickle
from torchtext import data


class moltokenize:
    def __init__(self, add_sep=False):
        if add_sep:
            self.tokenizer = self._tokenizer_with_sep
        else:
            self.tokenizer = self._tokenizer
        generaral_pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        self.regex = re.compile(generaral_pattern)

    def _tokenize(self, sentence):
        return [token for token in self.regex.findall(sentence)]

    def _tokenizer(self, sentence):
        return [tok for tok in self._tokenize(sentence) if tok != " "]

    def _tokenizer_with_sep(self, sentence):
        pattern = re.compile(r'(<sep>)')
        res = pattern.split(sentence)
        if len(res) == 1:
            return self._tokenizer(sentence) # no <sep>
        elif len(res) == 3:
            return self._tokenize(res[0]) + ['<sep>'] + self._tokenize(res[2])
        else:
            return []

    @staticmethod
    def untokenizer(tokens, sos_idx, eos_idx, itos):
        smi = ""
        for token in tokens:
            if token == eos_idx:
                break
            elif token != sos_idx:
                smi += itos[token]
        return smi
    

def smiles_field(field_path=None, add_sep=False):
    t_src = moltokenize(add_sep)
    t_trg = moltokenize(add_sep)
    
    SRC = data.Field(
        tokenize=t_src.tokenizer,
        batch_first=True,
        unk_token='<unk>'
    )
    
    TRG = data.Field(
        tokenize=t_trg.tokenizer,
        batch_first=True,
        init_token='<sos>',
        eos_token='<eos>',
        unk_token='<unk>'
    )
    
    if field_path is not None:
        suffix = '_sep' if add_sep else ''
        try:
            SRC = pickle.load(open(os.path.join(field_path, f'SRC{suffix}.pkl'), 'rb'))
            TRG = pickle.load(open(os.path.join(field_path, f'TRG{suffix}.pkl'), 'rb'))
        except:
            print(">>> Files SRC.pkl/TRG.pkl not in: " + os.path.join(field_path, f'SRC{suffix}.pkl'))
            exit(1)

    return (SRC, TRG)
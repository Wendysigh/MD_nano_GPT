from tokenizers import Tokenizer
from tokenizers.models import BPE
import numpy as np
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

from tokenizers.pre_tokenizers import Whitespace
tokenizer.pre_tokenizer = Whitespace()


train_txt = np.loadtxt('/home/wzengad/projects/MD_code/data/RMSD/train',dtype=str).reshape(-1)

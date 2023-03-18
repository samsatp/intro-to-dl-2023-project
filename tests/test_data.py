from data import collate_fn
import torch
from torch.nn.utils.rnn import PackedSequence
from data import Tokenizer, MultiLabelDataset
from typing import List

def test_collate_function():
    sample_x = [
        torch.tensor([1, 2, 3]),
        torch.tensor([4, 5, 6, 7, 8]),
        torch.tensor([9])
    ]
    sample_y = [
        torch.tensor([0,1,0]),
        torch.tensor([1,0,1]),
        torch.tensor([1,1,1])
    ]
    xs, ys = collate_fn((sample_x, sample_y))

    assert isinstance(xs, PackedSequence)


class TrivialTokenizer(Tokenizer):
    def __call__(self, text: str) -> List[list]:
        return text.split()
    
def test_getitem(get_data):
    headlines, texts, labels = get_data
    tokenizer = TrivialTokenizer()
    dataset = MultiLabelDataset(texts, labels, tokenizer)
    x, y = dataset[2]

    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)

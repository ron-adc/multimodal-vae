import os
import json
import tempfile
import pytest
import torch
from src.data import build_vocab, MultimodalDataset


def test_build_vocab_basic():
    captions = ["a cat on a mat", "a dog"]
    word2idx, idx2word = build_vocab(captions, freq_threshold=1)
    # Expect <PAD> token at index 0
    assert word2idx.get('<PAD>') == 0
    # 'a' should be in vocab
    assert word2idx.get('a') is not None


def test_dataset_len_and_getitem(tmp_path):
    # Create dummy image file
    img_path = tmp_path / "dummy.jpg"
    # Create a 64x64 black image
    from PIL import Image
    Image.new('RGB', (64, 64)).save(img_path)

    # Create captions JSON in COCO format
    ann = {
        "images": [{"id": 1, "file_name": "dummy.jpg"}],
        "annotations": [{"image_id": 1, "caption": "hello world"}]
    }
    ann_path = tmp_path / "annotations.json"
    with open(ann_path, 'w') as f:
        json.dump(ann, f)

    dataset = MultimodalDataset(str(tmp_path), str(ann_path), freq_threshold=1)
    assert len(dataset) == 1

    img, caption_idx = dataset[0]
    assert isinstance(img, torch.Tensor)
    assert isinstance(caption_idx, torch.Tensor)

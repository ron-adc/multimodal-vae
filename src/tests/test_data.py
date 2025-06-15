import pytest
import torch
import json
from torchvision import transforms
from PIL import Image
from src.data import MultimodalDataset, build_vocab

@pytest.fixture
def sample_data(tmp_path):
    # Create temporary image directory and JSON file
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    captions_filepath = tmp_path / "captions.json"

    # Create sample images
    for i in range(3):
        img = Image.new("RGB", (64, 64), color=(i * 50, i * 50, i * 50))
        img.save(img_dir / f"image_{i}.jpg")

    # Create sample captions file
    captions_data = {
        "images": [
            {"id": 0, "file_name": "image_0.jpg"},
            {"id": 1, "file_name": "image_1.jpg"},
            {"id": 2, "file_name": "image_2.jpg"},
        ],
        "annotations": [
            {"image_id": 0, "caption": "A red image."},
            {"image_id": 1, "caption": "A green image."},
            {"image_id": 2, "caption": "A blue image."},
        ],
    }
    with open(captions_filepath, "w") as f:
        json.dump(captions_data, f)

    return img_dir, captions_filepath


def test_build_vocab():
    captions = ["A red image.", "A green image.", "A blue image."]
    word2idx, idx2word = build_vocab(captions, freq_threshold=1)

    assert "<PAD>" in word2idx
    assert "<SOS>" in word2idx
    assert "<EOS>" in word2idx
    assert "<UNK>" in word2idx
    assert "red" in word2idx
    assert "green" in word2idx
    assert "blue" in word2idx

    word = "red"
    index = word2idx[word]
    assert idx2word[index] == word, f"Expected idx2word[{index}] to be '{word}', got '{idx2word[index]}'"     


def test_multimodal_dataset_length(sample_data):
    img_dir, captions_filepath = sample_data
    dataset = MultimodalDataset(img_dir, captions_filepath)

    assert len(dataset) == 3, f"Expected dataset length 3, got {len(dataset)}"


def test_multimodal_dataset_item(sample_data):
    img_dir, captions_filepath = sample_data
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = MultimodalDataset(img_dir, captions_filepath, transform=transform)

    image, caption_tensor = dataset[0]

    assert isinstance(image, torch.Tensor), "Image is not a tensor"
    assert image.shape[0] == 3, "Image does not have 3 channels"
    assert isinstance(caption_tensor, torch.Tensor), "Caption is not a tensor"
    assert caption_tensor[0].item() == dataset.word2idx["<SOS>"], "Caption does not start with <SOS>"
    assert caption_tensor[-1].item() == dataset.word2idx["<EOS>"], "Caption does not end with <EOS>"


def test_multimodal_dataset_invalid_index(sample_data):
    img_dir, captions_filepath = sample_data
    dataset = MultimodalDataset(img_dir, captions_filepath)

    with pytest.raises(IndexError):
        _ = dataset[10]
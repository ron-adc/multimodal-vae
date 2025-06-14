import os
import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import torch

import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize


def build_vocab(captions: list[str], freq_threshold: int):
    """
    Build word2idx and idx2word mappings.

    Args:
        captions: List of caption strings.
        freq_threshold: Minimum frequency to include a word.

    Returns:
        word2idx (dict), idx2word (dict)
    """
    word_freq = {}
    for caption in captions:
        for word in word_tokenize(caption.lower()):
            word_freq[word] = word_freq.get(word, 0) + 1
    
    specials = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
    word2idx = {word: idx for idx, word in enumerate(specials)}
    idx2word = {idx: word for idx, word in enumerate(specials)}

    # Filter words by frequency threshold
    filtered_words = sorted([word for word, freq in word_freq.items() if freq >= freq_threshold])
    for idx, word in enumerate(filtered_words, start=len(specials)):
        word2idx[word] = idx
        idx2word[idx] = word
    
    return word2idx, idx2word


class MultimodalDataset(Dataset):
    def __init__(
            self, 
            img_dir: str, 
            captions_filepath: str, 
            transform: transforms.Compose = None, 
            freq_threshold: int = 5):
        """
        Dataset for paired image-caption data.

        Args:
            img_dir: Directory with images.
            captions_file: Path to JSON annotations (COCO format).
            transform: torchvision transforms for images.
            freq_threshold: Vocabulary frequency threshold.
        """
        self.img_dir = img_dir
        self.freq_threshold = freq_threshold
        self.transform = transform or transforms.ToTensor()
        
        with open(captions_filepath, 'r') as f:
            captions_file = json.load(f)

        id2file = {img['id']: img['file_name'] for img in captions_file['images']}

        self.image_caption_pairs = []
        for ann in captions_file['annotations']:
            caption = ann['caption']
            filename = id2file.get(ann['image_id'])
            if filename:
                self.image_caption_pairs.append((filename, caption))
        captions = [c for _, c in self.image_caption_pairs]
        self.word2idx, self.idx2word = build_vocab(captions, freq_threshold)

    def __len__(self):
        return len(self.image_caption_pairs)

    def __getitem__(self, idx):
        filename, caption = self.image_caption_pairs[idx]

        # Load and transform image
        image = Image.open(os.path.join(self.img_dir, filename)).convert('RGB')
        image = self.transform(image)

        # Tokenize caption
        tokens = word_tokenize(caption.lower())
        caption_idx = [self.word2idx.get(token, self.word2idx['<UNK>']) for token in tokens]

        # Add <SOS> and <EOS> tokens
        sequence = [self.word2idx['<SOS>']] + caption_idx + [self.word2idx['<EOS>']]
        caption_tensor = torch.tensor(sequence, dtype=torch.long)

        return image, caption_tensor

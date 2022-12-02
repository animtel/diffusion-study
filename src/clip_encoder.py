from typing import Optional, Tuple, Sequence
import clip
import torch
from torch import nn
from PIL import Image


class OpenAICLIPTextEncoder(nn.Module):
    """Encode text into embeddings using the CLIP model."""

    def __init__(
        self,
        pretrained_model_name_or_path: str = 'ViT-L/14',
        max_length: int = 77,
        device: str = 'cpu',
        batch_size: int = 32,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.max_length = max_length

        self.device = device
        self.tokenizer = clip.tokenize
        self.model, _ = clip.load(pretrained_model_name_or_path, device=device)

    def forward(self, text):
        with torch.inference_mode():
            input_tokens = self._generate_input_tokens(text)
            embeddings = self.model.encode_text(input_tokens)
            return embeddings

    def _generate_input_tokens(self, texts: Sequence[str]):
        input_tokens = self.tokenizer(
            texts,
            context_length=self.max_length,
            truncate=True
        ).to(self.device)
        # input_tokens = {k: v.to(self.device) for k, v in input_tokens.items()}
        return input_tokens
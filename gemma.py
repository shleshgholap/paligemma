import torch
import torch.nn as nn
from typing import List, Optional, Union, Iterable, Tuple
from torch.nn import CrossEntropyLoss
import math
from siglip import SiglipVisionModelConfig, SiglipVisionModel


class PaliGemmaForConditionalGeneration(nn.Module):

    def __init__(self, config: PaliGemmaConfig):

        super(PaliGemmaForConditionalGeneration, self).__init__()
        self.config = config
        self.vision_model = SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.vocab_size

        language_model = GemmaForCausalLM(config.text_config)
        self.language_model = language_model

        self.pad_token_id = config.pad_token_id if config.pad_token_id is not None else -1

    def tie_weights(self):
        return self.language_model.tie_weights()

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            pixel_values: torch.FloatTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            kc_cache: Optional[KVCache] = None,
    ) -> Tuple:
        
        assert torch.all(attention_mask == 1), "The input cannot be padded for now"

        # 1. Extra the input embeddings
        # shape: (Batch_size, seq_len, hidden_size)
        input_embeddings = self.language_model.get_input_embeddings()(input_ids)
        

        pass
import tprch
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
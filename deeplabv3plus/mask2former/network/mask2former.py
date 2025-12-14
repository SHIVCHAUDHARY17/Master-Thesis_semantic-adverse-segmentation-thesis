import torch
import torch.nn as nn
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerConfig


class Mask2FormerWithWASTAS(nn.Module):
    def __init__(self, num_classes=19):
        super().__init__()

        config = Mask2FormerConfig.from_pretrained(
            "facebook/mask2former-swin-large-cityscapes-semantic"
        )
        config.num_labels = num_classes
        config.output_auxiliary_logits = True
        config.output_hidden_states = True
        config.semantic_loss_ignore_index = 255
        
        # Create ID to label mapping for Cityscapes
        config.id2label = {
            0: "road", 1: "sidewalk", 2: "building", 3: "wall", 4: "fence",
            5: "pole", 6: "traffic_light", 7: "traffic_sign", 8: "vegetation",
            9: "terrain", 10: "sky", 11: "person", 12: "rider", 13: "car",
            14: "truck", 15: "bus", 16: "train", 17: "motorcycle", 18: "bicycle"
        }
        config.label2id = {v: k for k, v in config.id2label.items()}

        self.mask2former = Mask2FormerForUniversalSegmentation.from_pretrained(
            "facebook/mask2former-swin-large-cityscapes-semantic",
            config=config,
            ignore_mismatched_sizes=False
        )

        hidden_dim = self.mask2former.config.hidden_size
        self.weather_head = nn.Linear(hidden_dim, 4)  # clear, rain, fog, snow
        self.time_head = nn.Linear(hidden_dim, 2)     # day, night

    def forward(self, pixel_values):
        outputs = self.mask2former(pixel_values=pixel_values)
        
        # ✅ Use the main semantic logits directly (already [B, num_labels, H, W])
        seg_logits = outputs.logits
        
        # Get decoder hidden states for weather/time heads
        if hasattr(outputs, 'transformer_decoder_last_hidden_state'):
            decoder_hidden = outputs.transformer_decoder_last_hidden_state
        else:
            # Fallback: use the last hidden state
            decoder_hidden = outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else None
        
        if decoder_hidden is not None:
            pooled = decoder_hidden.mean(dim=1)
            weather_logits = self.weather_head(pooled)
            time_logits = self.time_head(pooled)
        else:
            # Create dummy outputs if decoder hidden state is unavailable
            batch_size = pixel_values.size(0)
            weather_logits = torch.zeros(batch_size, 4, device=pixel_values.device)
            time_logits = torch.zeros(batch_size, 2, device=pixel_values.device)

        return seg_logits, weather_logits, time_logits

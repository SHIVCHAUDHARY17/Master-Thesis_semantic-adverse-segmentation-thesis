import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from transformers import Mask2FormerForUniversalSegmentation

class Mask2FormerWithWASTAS(nn.Module):
    def __init__(
        self,
        num_classes: int = 19,
        *,
        pretrained_card: str = "facebook/mask2former-swin-large-cityscapes-semantic",
        was_classes: int = 4,   # clear, rain, fog, snow
        tas_classes: int = 2,   # day, night
        output_hidden_states: bool = True,
        output_auxiliary_logits: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes

        # HuggingFace Mask2Former (Cityscapes card)
        self.mask2former = Mask2FormerForUniversalSegmentation.from_pretrained(
            pretrained_card,
            output_hidden_states=output_hidden_states,
            output_auxiliary_logits=output_auxiliary_logits,
        )
        print("backbone model_type:",
              getattr(self.mask2former.config.backbone_config, "model_type", None))
        print("backbone class:",
              self.mask2former.model.pixel_level_module.encoder.__class__.__name__)

        # Decoder/transformer hidden size
        self.hidden_dim = self.mask2former.config.hidden_size  # e.g., 256

        # --- small, DeepLab-like MLP trunk for WAS/TAS ---
        self.norm_aux = nn.LayerNorm(self.hidden_dim)  # stabilizes pooled token stats
        self.aux_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 84),              nn.ReLU(inplace=True),
        )
        self.weather_head = nn.Linear(84, was_classes)
        self.time_head    = nn.Linear(84, tas_classes)

        # Projection fallback when we must pool class scores / dense logits
        self.aux_proj_from_classes = nn.Linear(num_classes, self.hidden_dim)

        # Lazy adapter to map encoder feature dim -> hidden_dim when they differ
        self.aux_in_proj: Optional[nn.Linear] = None
        self._printed_aux_shapes = False  # one-time debug

    def _build_dense_logits_from_queries(self, outputs, pixel_values: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Build per-pixel logits from query class scores and masks if outputs.logits is not present.
        """
        class_q = getattr(outputs, "class_queries_logits", None)   # [B,Q,C+1]
        mask_q  = getattr(outputs, "masks_queries_logits", None)   # [B,Q,H',W']
        if class_q is None or mask_q is None:
            return None

        class_scores = class_q.softmax(dim=-1)[..., : self.num_classes]   # [B,Q,C]
        mask_probs   = mask_q.sigmoid()                                   # [B,Q,H',W']
        mask_probs   = F.interpolate(mask_probs, size=pixel_values.shape[-2:], mode="bilinear", align_corners=False)
        # aggregate queries -> dense [B,C,H,W]
        return torch.einsum("bqc,bqhw->bchw", class_scores, mask_probs)

    def _global_pool_encoder(self, x: torch.Tensor) -> torch.Tensor:
        """
        Robust global pooling for encoder outputs:
          - 4D: [B,C,H,W] or [B,H,W,C] -> [B,C]
          - 3D: [B,N,C] or [B,C,N]     -> [B,C]
          - 2D: [B,C]                  -> [B,C]
        """
        if x.dim() == 4:
            # assume either [B,C,H,W] or [B,H,W,C]
            if x.shape[1] <= 8 and x.shape[-1] > 8:
                # likely [B,H,W,C] -> [B,C,H,W]
                x = x.permute(0, 3, 1, 2).contiguous()
            # GAP over H,W
            x = x.mean(dim=(2, 3))  # [B,C]
            return x

        if x.dim() == 3:
            # could be [B,N,C] or [B,C,N]
            if x.shape[-1] <= 8 and x.shape[1] > 8:
                # [B,C,N] -> pool over N
                return x.mean(dim=2)  # [B,C]
            # default: [B,N,C] -> pool over N
            return x.mean(dim=1)      # [B,C]

        if x.dim() == 2:
            return x  # [B,C]

        raise RuntimeError(f"Unexpected encoder feature shape: {tuple(x.shape)}")

    def _adapt_aux_in(self, pooled: torch.Tensor) -> torch.Tensor:
        """
        Map pooled encoder features to self.hidden_dim if needed (lazy init)
        and ensure the adapter lives on the same device/dtype as `pooled`.
        """
        B, Cin = pooled.shape

        # Create the adapter if needed
        if Cin != self.hidden_dim and self.aux_in_proj is None:
            print(f"[aux-adapter] creating Linear({Cin} -> {self.hidden_dim}) for WS/TAS")
            self.aux_in_proj = nn.Linear(Cin, self.hidden_dim)

        # If we have an adapter, ensure it’s on the same device/dtype as `pooled`
        if self.aux_in_proj is not None:
            # Move only if needed (keeps it cheap)
            if self.aux_in_proj.weight.device != pooled.device or self.aux_in_proj.weight.dtype != pooled.dtype:
                self.aux_in_proj = self.aux_in_proj.to(device=pooled.device, dtype=pooled.dtype)

            pooled = self.aux_in_proj(pooled)

        return pooled

    def _pool_for_aux(self, outputs, seg_logits: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Prefer **encoder** features (to route WS/TAS grads into the encoder) — DeepLab-style.
        """
        enc_last = getattr(outputs, "encoder_last_hidden_state", None)   # often [B, N_tokens, C_enc] or [B,C,H,W]
        if torch.is_tensor(enc_last):
            pooled = self._global_pool_encoder(enc_last)                 # [B, C_enc]
            pooled = self._adapt_aux_in(pooled)                          # [B, hidden_dim]
            if not self._printed_aux_shapes:
                print(f"[aux] encoder pooled -> {tuple(pooled.shape)} (hidden_dim={self.hidden_dim})")
                self._printed_aux_shapes = True
            return pooled

        # Fallbacks if encoder states aren’t returned (shouldn’t happen with output_hidden_states=True)
        dec_last = getattr(outputs, "transformer_decoder_last_hidden_state", None)  # [B, Q, hidden_dim]
        if torch.is_tensor(dec_last):
            pooled = dec_last.mean(dim=1)                                # [B, hidden_dim]
            return pooled

        class_q = getattr(outputs, "class_queries_logits", None)         # [B,Q,C+1]
        if torch.is_tensor(class_q):
            cls_scores = class_q.softmax(dim=-1)[..., : self.num_classes].mean(dim=1)  # [B,C]
            return self.aux_proj_from_classes(cls_scores)                 # -> [B, hidden_dim]

        if torch.is_tensor(seg_logits):
            pooled_c = seg_logits.mean(dim=(2,3))                         # [B,C]
            return self.aux_proj_from_classes(pooled_c)                   # -> [B, hidden_dim]

        raise RuntimeError("Could not derive pooled features for WAS/TAS heads.")

    def forward(self, pixel_values: Optional[torch.Tensor] = None, *args, **kwargs):
        # allow positional call: model(images)
        if pixel_values is None and len(args) >= 1 and torch.is_tensor(args[0]):
            pixel_values = args[0]

        outputs = self.mask2former(pixel_values=pixel_values)

        # segmentation logits (dense)
        seg_logits = getattr(outputs, "logits", None)
        if seg_logits is None:
            seg_logits = self._build_dense_logits_from_queries(outputs, pixel_values)
        if seg_logits is None:
            aux = getattr(outputs, "auxiliary_logits", None)
            if isinstance(aux, (list, tuple)) and len(aux) > 0 and torch.is_tensor(aux[-1]):
                seg_logits = aux[-1]
            else:
                raise ValueError("Mask2Former outputs contain neither logits nor queries/aux logits.")

        # WS/TAS pooled features from **encoder** (with dim adaptation)
        pooled = self._pool_for_aux(outputs, seg_logits)     # [B, hidden_dim]
        pooled = self.norm_aux(pooled)
        h = self.aux_mlp(pooled)
        weather_logits = self.weather_head(h)
        time_logits    = self.time_head(h)

        return seg_logits, weather_logits, time_logits


from typing import List, Optional, Tuple, TypedDict
import warnings
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import Cache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter

from surya.common.pretrained import SuryaPreTrainedModel
from surya.common.s3 import S3DownloaderMixin
from surya.common.surya.config import SuryaModelConfig
from surya.common.surya.decoder import SuryaDecoderModel
from surya.common.surya.embedder import SimpleTokenEmbedder
from surya.common.surya.encoder import SuryaEncoderModel
from surya.settings import settings

from surya.logging import get_logger

logger = get_logger()


@dataclass
class SuryaModelOutput(CausalLMOutputWithPast):
    bbox_logits: torch.FloatTensor = None
    lm_logits: torch.FloatTensor = None


class FlashAttentionKwargs(TypedDict, total=False):
    """
    Keyword arguments for Flash Attention with Compile.

    Attributes:
        cu_seq_lens_q (`torch.LongTensor`, *optional*)
            Gets cumlative sequence length for query state.
        cu_seq_lens_k (`torch.LongTensor`, *optional*)
            Gets cumlative sequence length for key state.
        max_length_q (`int`, *optional*):
            Maximum sequence length for query state.
        max_length_k (`int`, *optional*):
            Maximum sequence length for key state.
    """

    cu_seq_lens_q: Optional[torch.LongTensor]
    cu_seq_lens_k: Optional[torch.LongTensor]
    max_length_q: Optional[int]
    max_length_k: Optional[int]


class KwargsForCausalLM(FlashAttentionKwargs): ...


class DistanceProjection(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(out_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)


class SuryaModel(S3DownloaderMixin, SuryaPreTrainedModel):
    config_class = SuryaModelConfig
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True
    main_input_name = "input_ids"
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(
        self,
        config: SuryaModelConfig,
        embedder: SimpleTokenEmbedder = None,
        vision_encoder: SuryaEncoderModel = None,
        decoder: SuryaDecoderModel = None,
        **kwargs,
    ):
        super().__init__(config, **kwargs)

        if vision_encoder is None:
            vision_encoder = SuryaEncoderModel(config.vision_encoder)

        if decoder is None:
            decoder = SuryaDecoderModel(config.decoder)

        if embedder is None:
            embedder = SimpleTokenEmbedder(config)

        self.vision_encoder = vision_encoder
        self.decoder = decoder
        self.embedder = embedder
        self.encoder_seq_len = getattr(config, "encoder_seq_len", None)

        # Simple encoding for image patches
        self.img_w_embed = nn.Embedding(
            self.config.image_embed_encoding_size,
            self.config.hidden_size,
        )

        self.img_h_embed = nn.Embedding(
            self.config.image_embed_encoding_size,
            self.config.hidden_size,
        )

        # Tying configs
        self.vision_encoder.config = self.config.vision_encoder
        self.decoder.config = self.config.decoder

        self.bbox_head = nn.Linear(config.hidden_size, 6)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

        if (
            self.config.multi_output_distance is not None
            and self.config.multi_output_distance > 0
        ):
            self.multi_output_projections = nn.ModuleList(
                [
                    DistanceProjection(
                        in_features=config.hidden_size, out_features=config.hidden_size
                    )
                    for _ in range(self.config.multi_output_distance)
                ]
            )

    def tie_weights(self):
        self._tie_weights()

    def _tie_weights(self):
        # Tie weights of lm head and token embedder
        self._tie_or_clone_weights(self.lm_head, self.embedder.token_embed)

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def get_input_embeddings(self) -> nn.Module:
        return self.embedder.token_embed

    def set_output_embeddings(self, new_embeddings: nn.Module):
        self.lm_head = new_embeddings

    def set_input_embeddings(self, new_embeddings: nn.Module):
        self.embedder.token_embed = new_embeddings

    def _unpack_image_embeddings_with_mask(
        self,
        image_embeds_flat: torch.Tensor,
        grid_thw: torch.Tensor,
        decoder_seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Unpacks image_embeds into batch-major form and constructs an additive attention mask.

        Args:
            image_embeds_flat: Tensor of shape (total_seq_len, embed_dim)
            grid_thw: Tensor of shape (batch_size, 3) with grid dimensions
            decoder_seq_len: Length of decoder sequence

        Returns:
            batched_image_embeds: Tensor of shape (batch_size, max_seq_len, embed_dim)
            attention_mask: Tensor of shape (batch_size, 1, decoder_seq_len, max_seq_len)
                            with 0 for valid tokens and -inf for padding (for additive attention)
            seq_lengths: Tensor of shape (batch_size,) containing the valid encoder lengths
        """
        device = image_embeds_flat.device
        dtype = image_embeds_flat.dtype

        batch_size = grid_thw.shape[0]
        total_seq_len, embed_dim = image_embeds_flat.shape

        # Calculate sequence lengths for each batch item
        seq_lengths = grid_thw.prod(dim=-1) // (
            self.vision_encoder.spatial_merge_size**2
        )
        max_seq_len = seq_lengths.max().item()
        target_seq_len = max_seq_len
        if self.encoder_seq_len is not None:
            if max_seq_len > self.encoder_seq_len:
                raise ValueError(
                    "encoder_seq_len was set too low; increase it to fit image embeddings"
                )
            target_seq_len = self.encoder_seq_len

        # Build indices for scatter operation
        batch_indices = []
        position_indices = []

        for i, seq_len in enumerate(seq_lengths):
            batch_indices.extend([i] * seq_len)
            position_indices.extend(list(range(seq_len)))

        batch_indices = torch.tensor(batch_indices, device=device, dtype=torch.long)
        position_indices = torch.tensor(
            position_indices, device=device, dtype=torch.long
        )

        # Create batched embeddings tensor
        batched_image_embeds = torch.zeros(
            (batch_size, target_seq_len, embed_dim), device=device, dtype=dtype
        )

        # Create additive attention mask: shape (batch_size, decoder_seq_len, max_seq_len)
        # - Rows = queries (from decoder), Columns = keys (from encoder)
        # - -inf for padding, 0 for valid positions
        attention_mask = torch.full(
            (batch_size, decoder_seq_len, target_seq_len),
            fill_value=float('-inf'),
            device=device,
            dtype=dtype,
        )
        for b in range(batch_size):
            valid_len = seq_lengths[b].item()
            attention_mask[b, :, :valid_len] = 0  # Unmasked

        attention_mask = attention_mask.unsqueeze(1)  # (batch_size, 1, decoder_seq_len, max_seq_len)

        # Scatter flat embeddings into batched form
        batched_image_embeds[batch_indices, position_indices] = image_embeds_flat

        return batched_image_embeds, attention_mask, seq_lengths

    def maybe_static_pad_image_inputs(
        self,
        chunk_pixels: torch.Tensor,
        chunk_grid_thw: torch.Tensor,
        actual_chunk_len: int,
        encoder_chunk_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        valid_embed_len = actual_chunk_len // (
            self.vision_encoder.spatial_merge_size**2
        )
        if settings.FOUNDATION_STATIC_CACHE and actual_chunk_len < encoder_chunk_size:
            padding_len = encoder_chunk_size - actual_chunk_len
            padding = torch.zeros(
                padding_len,
                *chunk_pixels.shape[1:],
                device=chunk_pixels.device,
                dtype=chunk_pixels.dtype,
            )
            chunk_pixels = torch.cat([chunk_pixels, padding], dim=0)

            padding_grid = torch.tensor(
                [[1, 2, padding_len // 2]],
                device=chunk_grid_thw.device,
                dtype=chunk_grid_thw.dtype,
            )
            chunk_grid_thw = torch.cat([chunk_grid_thw, padding_grid], dim=0)

        return chunk_pixels, chunk_grid_thw, valid_embed_len

    def get_image_embeddings(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        encoder_chunk_size: int | None,
        decoder_seq_len: int,
    ):
        """
        Embed images and return batched embeddings with cross-attention mask.
        For cross-attention: returns (batch_size, max_tokens, hidden_size) and mask.
        """
        # embed all images with the vision encoder after they have already been tiled and flattened into a single batch
        chunks = [0]
        grid_chunks = [0]
        curr_chunk_len = 0
        curr_seq_len = 0
        for i in range(len(grid_thw)):
            curr_chunk_len += (grid_thw[i][0] * grid_thw[i][1] * grid_thw[i][2]).item()
            if curr_chunk_len > encoder_chunk_size:
                chunks.append(curr_chunk_len + curr_seq_len)
                curr_seq_len += curr_chunk_len
                curr_chunk_len = 0
                grid_chunks.append(i + 1)

        if curr_chunk_len > 0:
            chunks.append(pixel_values.shape[0])
            grid_chunks.append(len(grid_thw))

        assert curr_chunk_len + curr_seq_len == pixel_values.shape[0], (
            f"Mismatch in encoder chunking, {curr_chunk_len} + {curr_seq_len} != {pixel_values.shape[0]}"
        )

        logger.debug(
            f"Chunking encoder sequence into {len(chunks) - 1} chunks of size {encoder_chunk_size} with lengths {chunks} and grids {grid_chunks}"
        )
        embeddings = []
        for i in range(len(chunks) - 1):
            start = chunks[i]
            end = chunks[i + 1]
            grid_start = grid_chunks[i]
            grid_end = grid_chunks[i + 1]

            chunk_pixels = pixel_values[start:end]
            chunk_grid_thw = grid_thw[grid_start:grid_end]
            actual_chunk_len = end - start
            chunk_pixels, chunk_grid_thw, valid_embed_len = (
                self.maybe_static_pad_image_inputs(
                    chunk_pixels, chunk_grid_thw, actual_chunk_len, encoder_chunk_size
                )
            )

            chunk_embeddings = self.vision_encoder.embed_images(
                image_batch=chunk_pixels, grid_thw=chunk_grid_thw
            )
            embeddings.append(chunk_embeddings[:valid_embed_len])

        if len(embeddings) == 0:
            raise ValueError(
                "No image embeddings were generated. Check the input images and grid sizes."
            )
        elif len(embeddings) == 1:
            embeddings = embeddings[0]
        else:
            embeddings = torch.cat(embeddings, dim=0)

        encoding_2d = self.get_2d_learned_embeddings(
            grid_thw,
            device=embeddings.device,
            bbox_size=self.config.image_embed_encoding_multiplier,
        )
        assert embeddings.shape[0] == encoding_2d.shape[0], (
            f"Mismatch in image embedding seq len: {embeddings.shape} vs {encoding_2d.shape}"
        )
        assert embeddings.shape[1] == encoding_2d.shape[1], (
            f"Mismatch in image embedding token counts: {embeddings.shape} vs {encoding_2d.shape}"
        )

        embeddings = embeddings + encoding_2d

        # For cross-attention: batch the flat embeddings and create attention mask
        (
            batched_embeddings,
            attention_mask,
            seq_lengths,
        ) = self._unpack_image_embeddings_with_mask(
            image_embeds_flat=embeddings,
            grid_thw=grid_thw,
            decoder_seq_len=decoder_seq_len,
        )
        attention_mask = attention_mask.to(dtype=embeddings.dtype)

        return batched_embeddings, attention_mask, seq_lengths

    def embed_ids_boxes_images(
        self, input_ids, pixel_values, grid_thw, encoder_chunk_size: int
    ):
        """
        For cross-attention model: embed input tokens and separately prepare image features
        for encoder embeddings (NOT inserted into the sequence)
        """
        # This is batched in the inner call
        inputs_embeds = self.embedder.embed(input_tokens=input_ids)
        encoder_embeddings = None
        encoder_attention_mask = None
        encoder_seq_lengths: Optional[torch.Tensor] = None

        # Assert no image tokens in the sequence (cross-attention model)
        assert (input_ids == self.config.image_token_id).sum() == 0, (
            "This is a cross-attention model. There should be no image token ids in the sequence!"
        )

        if pixel_values is not None:
            # get_image_embeddings now returns batched embeddings and mask
            (
                encoder_embeddings,
                encoder_attention_mask,
                encoder_seq_lengths,
            ) = self.get_image_embeddings(
                pixel_values=pixel_values,
                grid_thw=grid_thw,
                encoder_chunk_size=encoder_chunk_size,
                decoder_seq_len=input_ids.shape[1],
            )
            encoder_embeddings = encoder_embeddings.to(inputs_embeds.dtype)
            if encoder_attention_mask is not None:
                encoder_attention_mask = encoder_attention_mask.to(
                    dtype=inputs_embeds.dtype
                )

        return (
            inputs_embeds,
            encoder_embeddings,
            encoder_attention_mask,
            encoder_seq_lengths,
        )

    def _build_encoder_attention_inputs(
        self,
        image_features: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if image_features is None or image_features.numel() == 0:
            return None, None

        batch_size = input_ids.size(0)
        hidden_size = image_features.size(-1)
        token_counts = (input_ids == self.config.image_token_id).sum(dim=1)
        max_tokens = token_counts.max().item() if token_counts.numel() > 0 else 0

        if max_tokens == 0:
            return None, None

        total_tokens = token_counts.sum().item()
        if total_tokens != image_features.shape[0]:
            logger.warning(
                "Mismatch between image features and token counts while building cross-attention inputs"
            )
            return None, None

        encoder_embeddings = image_features.new_zeros((batch_size, max_tokens, hidden_size))

        offset = 0
        for batch_idx in range(batch_size):
            count = int(token_counts[batch_idx].item())
            if count == 0:
                continue
            next_offset = offset + count
            encoder_embeddings[batch_idx, :count] = image_features[offset:next_offset]
            offset = next_offset

        return encoder_embeddings, token_counts

    def _build_cross_attention_mask(
        self,
        encoder_embeddings: torch.Tensor,
        decoder_seq_len: int,
        encoder_seq_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Build cross-attention mask for encoder-decoder attention, masking padded encoder tokens.

        Args:
            encoder_embeddings: shape (batch_size, encoder_len, hidden_size)
            decoder_seq_len: length of decoder sequence (1 during generation)
            encoder_seq_lengths: optional tensor of shape (batch_size,) with valid encoder lengths

        Returns:
            attention_mask: shape (batch_size, 1, decoder_seq_len, encoder_len)
                            with 0 for valid tokens and -inf for padding positions
        """
        batch_size, encoder_len, _ = encoder_embeddings.shape
        device = encoder_embeddings.device
        dtype = encoder_embeddings.dtype

        if encoder_seq_lengths is None:
            encoder_seq_lengths = torch.full(
                (batch_size,),
                fill_value=encoder_len,
                device=device,
                dtype=torch.long,
            )
        else:
            encoder_seq_lengths = encoder_seq_lengths.to(
                device=device, dtype=torch.long
            )

        attention_mask = torch.full(
            (batch_size, decoder_seq_len, encoder_len),
            fill_value=float('-inf'),
            device=device,
            dtype=dtype,
        )
        for b in range(batch_size):
            valid_len = int(encoder_seq_lengths[b].item())
            if valid_len > 0:
                attention_mask[b, :, :valid_len] = 0.0

        return attention_mask.unsqueeze(1)

    def _get_cached_encoder_states(
        self,
        past_key_values,
        cache_idxs: Optional[List[int]],
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Retrieve cached encoder embeddings and valid lengths."""
        if past_key_values is None or cache_idxs is None:
            return None

        storage = getattr(past_key_values, "_encoder_states", None)
        if not storage:
            return None

        embeddings_list: List[Optional[torch.Tensor]] = []
        lengths: List[int] = []
        first_emb = None
        max_tokens = 0

        for cache_idx in cache_idxs:
            entry = storage.get(cache_idx)
            if entry is None:
                embeddings_list.append(None)
                lengths.append(0)
                continue

            if isinstance(entry, tuple):
                emb, valid_len = entry
            else:  # Backward compatibility with older cache format
                emb = entry
                valid_len = emb.shape[0]
            embeddings_list.append(emb)
            lengths.append(int(valid_len))
            if first_emb is None:
                first_emb = emb
            max_tokens = max(max_tokens, emb.shape[0])

        if first_emb is None:
            return None

        # Stack embeddings
        batch_size = len(cache_idxs)
        hidden_size = first_emb.shape[-1]
        device = first_emb.device
        dtype = first_emb.dtype

        stacked_embeddings = torch.zeros(
            (batch_size, max_tokens, hidden_size), device=device, dtype=dtype
        )

        for batch_idx, emb in enumerate(embeddings_list):
            if emb is None:
                continue
            stacked_embeddings[batch_idx, :emb.shape[0]] = emb
        lengths_tensor = torch.tensor(lengths, device=device, dtype=torch.long)

        return stacked_embeddings, lengths_tensor

    def _set_cached_encoder_states(
        self,
        past_key_values,
        cache_idxs: Optional[List[int]],
        encoder_embeddings: torch.Tensor | None,
        encoder_seq_lengths: Optional[torch.Tensor],
    ) -> None:
        """Store encoder embeddings and valid lengths in cache for reuse during decoding."""
        if past_key_values is None or cache_idxs is None:
            return

        if not hasattr(past_key_values, "_encoder_states"):
            past_key_values._encoder_states = {}

        storage = past_key_values._encoder_states

        if encoder_embeddings is None or encoder_seq_lengths is None:
            for cache_idx in cache_idxs:
                storage.pop(cache_idx, None)
            return

        batch_size = encoder_embeddings.shape[0]
        limit = min(len(cache_idxs), batch_size)

        for local_idx in range(limit):
            cache_idx = cache_idxs[local_idx]
            valid_len = int(encoder_seq_lengths[local_idx].item())
            if valid_len <= 0:
                storage[cache_idx] = None
                continue
            emb = encoder_embeddings[local_idx, :valid_len].detach().clone()
            storage[cache_idx] = (emb, valid_len)

        for cache_idx in cache_idxs[limit:]:
            storage.pop(cache_idx, None)

    def get_2d_learned_embeddings(
        self,
        grid_thw,
        device: str | torch.device = "cpu",
        bbox_size: int = 256,
    ):
        all_embeddings = []
        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h, llm_grid_w = (
                grid_h // self.config.merge_size,
                grid_w // self.config.merge_size,
            )

            # Scale to 0-1024
            llm_grid_h = (
                torch.arange(llm_grid_h, device=device)
                / max(1, (llm_grid_h - 1))
                * bbox_size
            )
            llm_grid_w = (
                torch.arange(llm_grid_w, device=device)
                / max(1, (llm_grid_w - 1))
                * bbox_size
            )

            llm_grid_w_idx = llm_grid_w.to(torch.long)
            llm_grid_h_idx = llm_grid_h.to(torch.long)

            llm_grid_w = self.img_w_embed(llm_grid_w_idx)
            llm_grid_h = self.img_h_embed(llm_grid_h_idx)

            full_grid = llm_grid_h[:, None] + llm_grid_w[None, :]

            flattened = full_grid.flatten(
                0, 1
            )  # Flatten first dimension, so they are seq_len x embed_dim
            all_embeddings.append(flattened)
        return torch.concat(
            all_embeddings, dim=0
        )  # Shape is num_image_tokens x embed_dim

    def get_logits(self, hidden_states):
        assert hidden_states.shape[1] == 1, (
            "Multi output predictions only applied on the last token"
        )

        all_lm_logits = []
        all_bbox_logits = []

        current_hidden = hidden_states

        # Loop includes initial prediction (i=0) plus multi_out_distance additional predictions
        for i in range(self.config.multi_output_distance + 1):
            if i > 0:
                current_hidden = self.multi_output_projections[i - 1](current_hidden)

            lm_logits = self.lm_head(current_hidden)
            bbox_logits = F.sigmoid(self.bbox_head(current_hidden))

            all_lm_logits.append(lm_logits)
            all_bbox_logits.append(bbox_logits)

        # Concatenate along sequence dimension (dim=1)
        final_lm_logits = torch.cat(all_lm_logits, dim=1)
        final_bbox_logits = torch.cat(all_bbox_logits, dim=1)

        return final_lm_logits, final_bbox_logits

    def forward(
        self,
        input_ids=None,
        labels=None,
        image_tiles=None,
        grid_thw=None,
        inputs_embeds=None,
        attention_mask=None,
        position_ids=None,
        cache_position=None,
        past_key_values=None,
        output_hidden_states=False,
        output_attentions=False,
        use_cache=False,
        encoder_chunk_size=32768,
        cache_idxs=None,
        num_valid_tokens=None,
        prefill=True,
        text_lengths=None,
        encoder_embeddings=None,
        encoder_attention_mask=None,
        logits_to_keep=None,
        **kwargs: KwargsForCausalLM,
    ):
        # Process the mixed batch if provided
        if any(
            [
                input_ids is None,
                (prefill and (image_tiles is None or grid_thw is None)),
                position_ids is None,
                cache_position is None,
            ]
        ):
            raise ValueError(
                "`input_ids`, `position_ids`, and `cache_position` **must** be specified. `image_tiles` and `grid_thw` are required for prefill"
            )

        (
            inputs_embeds,
            computed_encoder_embeddings,
            computed_encoder_attention_mask,
            computed_encoder_seq_lengths,
        ) = self.embed_ids_boxes_images(
            input_ids, image_tiles, grid_thw, encoder_chunk_size
        )

        encoder_seq_lengths = computed_encoder_seq_lengths

        # Use computed values if not provided
        if encoder_embeddings is None and computed_encoder_embeddings is not None:
            encoder_embeddings = computed_encoder_embeddings
        if encoder_attention_mask is None and computed_encoder_attention_mask is not None:
            encoder_attention_mask = computed_encoder_attention_mask
        if encoder_seq_lengths is None and computed_encoder_seq_lengths is not None:
            encoder_seq_lengths = computed_encoder_seq_lengths

        # Check cache for encoder states (for generation with KV cache)
        if encoder_embeddings is None and cache_idxs is not None:
            cached_encoder_state = self._get_cached_encoder_states(
                past_key_values, cache_idxs
            )
            if cached_encoder_state is not None:
                cached_embeddings, cached_lengths = cached_encoder_state
                encoder_embeddings = cached_embeddings
                encoder_seq_lengths = cached_lengths

        # Cache encoder embeddings for future use
        if computed_encoder_embeddings is not None:
            self._set_cached_encoder_states(
                past_key_values,
                cache_idxs,
                computed_encoder_embeddings,
                computed_encoder_seq_lengths,
            )

        # Build cross-attention mask with current decoder sequence length
        # This ensures mask shape matches during both prefill and generation
        if encoder_embeddings is not None and encoder_attention_mask is None:
            current_decoder_len = inputs_embeds.shape[1]
            encoder_attention_mask = self._build_cross_attention_mask(
                encoder_embeddings, current_decoder_len, encoder_seq_lengths
            )
        if (
            encoder_attention_mask is not None
            and encoder_attention_mask.dtype != inputs_embeds.dtype
        ):
            encoder_attention_mask = encoder_attention_mask.to(inputs_embeds.dtype)

        # Handling flash attention kwargs outside the decoder to speed up + avoid graph breaks inside the decoder
        # Skipped during decoding since not required
        if self.decoder.config._attn_implementation == "flash_attention_2" and prefill:
            # Needed for CPU -> GPU
            from surya.common.surya.flash_attn_utils import _get_unpad_data

            batch_size, query_length, _ = inputs_embeds.shape
            indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(
                attention_mask
            )
            kwargs["batch_size"] = batch_size
            kwargs["query_length"] = query_length
            kwargs["indices_k"] = indices_k
            kwargs["cu_seqlens_k"] = cu_seqlens_k
            kwargs["max_seqlen_in_batch_k"] = max_seqlen_in_batch_k

        causal_mask = self._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values,
            output_attentions,
        )

        attention_mask = causal_mask
        outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            past_key_values=past_key_values,
            encoder_embeddings=encoder_embeddings,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=True,
            use_cache=use_cache,
            cache_idxs=cache_idxs,
            num_valid_tokens=num_valid_tokens,
            prefill=prefill,
            text_lengths=text_lengths,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        if logits_to_keep is not None:
            hidden_states = hidden_states[:, -logits_to_keep:, :]
        hidden_states = hidden_states.contiguous()

        loss = None
        if labels is not None:
            # Training, return full logits
            lm_logits = self.lm_head(hidden_states)
            bbox_logits = None
            vocab_size = lm_logits.shape[-1]
            labels = torch.roll(labels, shifts=-1, dims=-1)
            loss = F.cross_entropy(
                lm_logits.view(-1, vocab_size), labels.view(-1), reduction="mean"
            )
        else:
            lm_logits, bbox_logits = self.get_logits(hidden_states)

        return SuryaModelOutput(
            loss=loss,
            bbox_logits=bbox_logits,
            lm_logits=lm_logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions if output_attentions else None,
            past_key_values=outputs.past_key_values,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.decoder.config._attn_implementation == "flash_attention_2":
            return attention_mask

        # We always pass in a 2D attention mask from the processor - In both static and dynamic cache cases
        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        target_length = (
            attention_mask.shape[-1]
            if isinstance(attention_mask, torch.Tensor)
            else past_key_values.max_cache_len
        )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
            config=self.config,
            past_key_values=past_key_values,
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu"]
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(
                causal_mask, min_dtype
            )

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        config: SuryaModelConfig,
        past_key_values: Cache,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to plcae the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence. Shape `(batch_size, sequence_length)`.
            batch_size (`torch.Tensor`):
                Batch size.
            config (`Qwen2Config`):
                The model's configuration class
            past_key_values (`Cache`):
                The cache class that is being used currently to generate
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length),
                fill_value=min_dtype,
                dtype=dtype,
                device=device,
            )
            # Batch-aware diagonal attend mask
            diagonal_attend_mask = torch.arange(target_length, device=device).unsqueeze(
                0
            ) > cache_position.unsqueeze(-1)
            causal_mask = (
                causal_mask.unsqueeze(0) * diagonal_attend_mask
            )  # (batch_size, seq_len, target_len)
            causal_mask = causal_mask[
                :, None, :, :
            ]  # (batch_size, 1, seq_len, target_len)
            if attention_mask is not None:
                causal_mask = (
                    causal_mask.clone()
                )  # copy to contiguous memory for in-place edit
                if attention_mask.shape[-1] > target_length:
                    attention_mask = attention_mask[:, :target_length]
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[
                    :, None, None, :
                ].to(causal_mask.device)
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[
                    :, :, :, :mask_length
                ].masked_fill(padding_mask, min_dtype)
        return causal_mask

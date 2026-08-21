"""
Rigorous tests for the KV-cache sequence-length invariant that
``Decoder.forward`` now *asserts* (models/decoder.py).

Background
----------
The HF ``Decoder`` does not thread ``start_pos`` into the backbone; positions are
derived by the HF model from the cache length. That is only safe if:

    the number of tokens the generation loop *thinks* it has produced
      ==  the number of slots the backbone actually stored in its Cache.

``Decoder.forward`` guards this on every decode step:

    if kv_cache is not None and start_pos != kv_cache.get_seq_length():
        raise ValueError(...)

The non-obvious risk is the VLM's image-token expansion: ``generate`` feeds
``token_embd`` whose length includes the (already expanded) image-token slots, and
``current_total_seq_len`` in the loop counts that full length. If a backbone ever
stored a number of cache slots different from the number of embeddings fed
(e.g. it collapsed image tokens), the loop's counter and the cache would desync and
the guard would fire. These tests prove the invariant holds end-to-end — including
the multi-token image path — for two real backbones with opposite architectures:

  * SmolLM2-360M : plain transformer, a growing key/value cache.
  * LFM2.5-230M  : hybrid attention + short-conv, a config-aware DynamicCache whose
                   conv layers do NOT grow with sequence length — exactly the case
                   where "one embedding in == one slot cached" is least obvious.

Design notes
------------
  * ``load_backbone=False`` everywhere -> random weights, no large downloads
    (only the small config.json + tokenizer, which are cached).
  * A ViT of img_size 64 / patch 16 yields a 4x4 patch grid -> pixel_shuffle
    factor 2 -> 4 image tokens per image, so the image expansion really is
    multi-token (not a degenerate single slot).
  * Generation is instrumented by wrapping ``decoder.forward`` to record, on every
    call, the sequence length fed in, the ``start_pos`` passed, and the cache length
    before/after. The invariant is then checked against those records.
"""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from models.config import VLMConfig
from models.vision_language_model import VisionLanguageModel


SMOL_ID: str = "HuggingFaceTB/SmolLM2-360M-Instruct"  # hidden 960, plain transformer
LFM_ID: str = "LiquidAI/LFM2.5-230M"                   # hidden 1024, hybrid conv+attn


def _cfg(model_id: str, hidden_dim: int) -> VLMConfig:
    """VLMConfig on the hf backend with a tiny ViT that expands to 4 image tokens/image."""
    return VLMConfig(
        lm_backend="hf",
        lm_model_type=model_id,
        lm_tokenizer=model_id,
        lm_hidden_dim=hidden_dim,
        # Tiny ViT: 64/16 = 4x4 = 16 patches; factor 2 -> 2x2 = 4 image tokens per image.
        vit_hidden_dim=48,
        vit_inter_dim=96,
        vit_n_heads=3,
        vit_n_blocks=1,
        vit_img_size=64,
        vit_patch_size=16,
        vit_dropout=0.0,
        mp_pixel_shuffle_factor=2,
    )


class TestKVCacheInvariant(unittest.TestCase):
    cfg_smol: VLMConfig
    vlm_smol: VisionLanguageModel
    cfg_lfm: VLMConfig
    vlm_lfm: VisionLanguageModel

    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(0)
        # .float() normalizes dtype: LFM2.5 loads in bf16 but resize_token_embeddings
        # produces a float32 embedding, leaving a mixed-dtype model. These tests are about
        # sequence-length accounting, not numerics, so cast everything to float32 to isolate
        # the invariant from that (separate) dtype concern.
        cls.cfg_smol = _cfg(SMOL_ID, hidden_dim=960)
        cls.vlm_smol = VisionLanguageModel(cls.cfg_smol, load_backbone=False).eval().float()
        cls.cfg_lfm = _cfg(LFM_ID, hidden_dim=1024)
        cls.vlm_lfm = VisionLanguageModel(cls.cfg_lfm, load_backbone=False).eval().float()

    def _models(self):
        return (("smol", self.vlm_smol, self.cfg_smol), ("lfm", self.vlm_lfm, self.cfg_lfm))

    # ---------------------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------------------
    def _prefill(self, vlm: VisionLanguageModel, seq_len: int):
        """Run one prefill of `seq_len` random embeddings through the decoder; return its Cache."""
        dim: int = vlm.cfg.lm_hidden_dim
        embd: torch.Tensor = torch.randn(1, seq_len, dim)
        with torch.no_grad():
            _, cache = vlm.decoder(embd, attention_mask=None, kv_cache=None, start_pos=0)
        return cache

    def _make_image_prompt(self, vlm: VisionLanguageModel, cfg: VLMConfig, n_text: int):
        """
        Build (input_ids, images) whose image-token count exactly matches the number of
        image embeddings the ViT+MP actually produces for one image. Returns
        (input_ids, images, n_img_tokens, total_len).
        """
        img: torch.Tensor = torch.randn(1, 3, cfg.vit_img_size, cfg.vit_img_size)
        with torch.no_grad():
            image_embd: torch.Tensor = vlm.MP(vlm.vision_encoder(img))  # [1, n_img_tokens, D]
        n_img_tokens: int = image_embd.size(1)

        image_token_id: int = vlm.tokenizer.image_token_id
        text_ids: list[int] = [5 + i for i in range(n_text)]
        prompt: list[int] = [image_token_id] * n_img_tokens + text_ids
        input_ids: torch.Tensor = torch.tensor([prompt], dtype=torch.long)  # [1, total]
        return input_ids, [img], n_img_tokens, len(prompt)

    def _generate_recording(self, vlm, input_ids, images, max_new_tokens):
        """
        Run generate() while recording, per decoder call:
          seq_in   : length of the embeddings fed this call,
          start_pos: the start_pos the loop passed,
          cin      : cache length going in  (0 if no cache),
          cout     : cache length coming out.
        Returns (generated_ids, records).
        """
        records: list[dict] = []
        original_forward = vlm.decoder.forward  # bound method, captured before patching

        def recording_forward(token_embd, attention_mask=None, kv_cache=None, start_pos=0):
            cin: int = kv_cache.get_seq_length() if kv_cache is not None else 0
            hidden, cache = original_forward(
                token_embd, attention_mask=attention_mask, kv_cache=kv_cache, start_pos=start_pos
            )
            cout = cache.get_seq_length() if cache is not None else None
            records.append(
                dict(seq_in=int(token_embd.size(1)), start_pos=int(start_pos), cin=int(cin), cout=cout)
            )
            return hidden, cache

        vlm.decoder.forward = recording_forward
        try:
            out = vlm.generate(input_ids, images, max_new_tokens=max_new_tokens, greedy=True)
        finally:
            vlm.decoder.forward = original_forward  # always restore
        return out, records

    def _assert_records_consistent(self, records, prompt_len, max_new_tokens):
        """The heart of the invariant: verify the loop's counter tracks the cache exactly."""
        # One prefill call + one call per generated token.
        self.assertEqual(len(records), 1 + max_new_tokens)

        prefill = records[0]
        # Prefill fed the whole prompt with no incoming cache...
        self.assertEqual(prefill["cin"], 0)
        self.assertEqual(prefill["start_pos"], 0)
        self.assertEqual(prefill["seq_in"], prompt_len)
        # ...and the cache stored exactly one slot per embedding fed (image tokens included).
        self.assertEqual(prefill["cout"], prompt_len)

        # Each decode step feeds exactly one token, at the right absolute position, and the
        # cache grows by exactly one — i.e. start_pos == incoming cache length, always.
        expected_cache_len = prompt_len
        for step in records[1:]:
            self.assertEqual(step["seq_in"], 1)
            self.assertEqual(step["cin"], expected_cache_len)
            self.assertEqual(step["start_pos"], expected_cache_len)  # the guarded invariant
            self.assertEqual(step["cout"], expected_cache_len + 1)
            expected_cache_len += 1

    # ---------------------------------------------------------------------------------
    # 1. Unit: the backbone caches exactly one slot per input embedding.
    # ---------------------------------------------------------------------------------
    def test_01_cache_stores_one_slot_per_embedding(self) -> None:
        for name, vlm, _ in self._models():
            with self.subTest(model=name):
                cache = self._prefill(vlm, seq_len=7)
                self.assertEqual(cache.get_seq_length(), 7)  # 7 embeddings -> 7 slots

                one: torch.Tensor = torch.randn(1, 1, vlm.cfg.lm_hidden_dim)
                with torch.no_grad():
                    _, cache = vlm.decoder(one, kv_cache=cache, start_pos=7)
                self.assertEqual(cache.get_seq_length(), 8)  # +1 token -> +1 slot

    # ---------------------------------------------------------------------------------
    # 2. Unit: the guard actually fires on desync, and stays silent when consistent.
    # ---------------------------------------------------------------------------------
    def test_02_guard_raises_on_startpos_cache_desync(self) -> None:
        for name, vlm, _ in self._models():
            with self.subTest(model=name):
                one: torch.Tensor = torch.randn(1, 1, vlm.cfg.lm_hidden_dim)

                # Correct start_pos (== cache length) must NOT raise.
                cache_ok = self._prefill(vlm, seq_len=5)
                with torch.no_grad():
                    vlm.decoder(one, kv_cache=cache_ok, start_pos=5)  # no error

                # Wrong start_pos (cache length is 5) must raise a clear ValueError.
                cache_bad = self._prefill(vlm, seq_len=5)
                with self.assertRaises(ValueError):
                    with torch.no_grad():
                        vlm.decoder(one, kv_cache=cache_bad, start_pos=4)

    # ---------------------------------------------------------------------------------
    # 3. Integration: text-only generation keeps loop counter == cache length.
    # ---------------------------------------------------------------------------------
    def test_03_generate_text_only_invariant(self) -> None:
        max_new_tokens: int = 4
        for name, vlm, _ in self._models():
            with self.subTest(model=name):
                input_ids: torch.Tensor = torch.tensor([[5, 6, 7, 8, 9]], dtype=torch.long)  # [1, 5]
                prompt_len: int = input_ids.size(1)
                _, records = self._generate_recording(vlm, input_ids, None, max_new_tokens)
                self._assert_records_consistent(records, prompt_len, max_new_tokens)

    # ---------------------------------------------------------------------------------
    # 4. Integration (the real concern): image-token expansion does NOT desync the
    #    loop counter from the cache. Each of the several image-token slots must occupy
    #    exactly one cache slot, so prefill cache length == full input length.
    # ---------------------------------------------------------------------------------
    def test_04_generate_with_images_seqlen_matches_cache(self) -> None:
        max_new_tokens: int = 4
        for name, vlm, cfg in self._models():
            with self.subTest(model=name):
                input_ids, images, n_img_tokens, total_len = self._make_image_prompt(vlm, cfg, n_text=3)

                # Sanity: the expansion really is multi-token, and the prompt length is
                # image tokens + text (this is exactly "what the loop counts").
                self.assertGreaterEqual(n_img_tokens, 2)
                self.assertEqual(total_len, n_img_tokens + 3)
                self.assertEqual(input_ids.size(1), total_len)

                out, records = self._generate_recording(vlm, input_ids, images, max_new_tokens)

                # Generation produced the requested number of tokens without the guard firing.
                self.assertEqual(out.shape, (1, max_new_tokens))
                # And the full invariant holds, with the image-inclusive prompt length as the base.
                self._assert_records_consistent(records, total_len, max_new_tokens)

    # ---------------------------------------------------------------------------------
    # 5. Direct check that image expansion preserves sequence length (the root cause the
    #    invariant depends on): #embeddings after replacement == #input ids.
    # ---------------------------------------------------------------------------------
    def test_05_image_replacement_preserves_sequence_length(self) -> None:
        for name, vlm, cfg in self._models():
            with self.subTest(model=name):
                input_ids, images, n_img_tokens, total_len = self._make_image_prompt(vlm, cfg, n_text=3)

                token_embd: torch.Tensor = vlm.decoder.token_embedding(input_ids)  # [1, total, D]
                with torch.no_grad():
                    image_embd = vlm.MP(vlm.vision_encoder(images[0]))
                    merged = vlm._replace_img_tokens_with_embd(input_ids, token_embd, image_embd)

                # Replacing image placeholders swaps values in place; it must not change length.
                self.assertEqual(merged.size(1), input_ids.size(1))
                self.assertEqual(merged.size(1), total_len)
                # The number of image-token slots equals the number of image embeddings supplied.
                mask = (input_ids == vlm.tokenizer.image_token_id)
                self.assertEqual(int(mask.sum()), n_img_tokens)


if __name__ == "__main__":
    unittest.main()

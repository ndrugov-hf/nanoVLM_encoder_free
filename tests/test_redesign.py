"""
Rigorous tests for the recent changes to ``models/vision_language_model.py``.

What "recent changes" means here (all in ``VisionLanguageModel.__init__``):

  1. Decoder construction was routed through ``build_decoder(cfg, load_backbone)``
     instead of hard-coding ``LanguageModel``. ``cfg.lm_backend`` now selects the
     backend: "custom" -> hand-written ``LanguageModel``; "hf" -> ``Decoder`` wrapper
     around any HuggingFace ``AutoModelForCausalLM`` (e.g. SmolLM2, LFM2.5).

  2. The tokenizer is now built *first*, before the decoder, because the decoder's
     embedding table must be reconciled against it.

  3. For the "hf" backend only, the decoder's embedding table is resized to
     ``len(tokenizer)`` and the rows for the 66 VLM special tokens are (re)initialized
     by ``_fit_embeddings_to_tokenizer``. This is the fix that lets the HF decoder look
     up the VLM's image / row-col tokens without an out-of-range indexing crash.

The tests below check each of those behaviors for two real HF decoders that stress the
two opposite resize regimes:
  * SmolLM2-360M  -> embedding table == base vocab, so resize GROWS it.
  * LFM2.5-230M   -> embedding table is padded larger than the vocab, so resize SHRINKS it.

Design notes:
  * ``load_backbone=False`` everywhere -> models are randomly initialized, so no large
    weight files are downloaded (only the small config.json + tokenizer, which are cached).
  * Heavy models are built once in ``setUpClass`` and shared across tests.
  * Every local variable is type-annotated to make it clear what it holds.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn
from transformers import AutoConfig

from models.config import VLMConfig
from models.vision_language_model import VisionLanguageModel
from models.decoder import Decoder
from models.language_model import LanguageModel
from models.build_decoder import build_decoder


# --- IDs of the two real HF decoders we test against --------------------------------
SMOL_ID: str = "HuggingFaceTB/SmolLM2-360M-Instruct"  # hidden_size 960, vocab == table -> GROW
LFM_ID: str = "LiquidAI/LFM2.5-230M"                   # hidden_size 1024, table padded  -> SHRINK


def _tiny_vit_kwargs() -> dict:
    """
    Return ViT config fields shrunk to almost nothing.

    We never actually run the vision encoder in these tests (all forward passes are
    text-only, and image replacement is tested with fabricated embeddings), so a tiny
    ViT keeps model construction cheap while still letting ``ModalityProjector`` build.
    """
    return dict(
        vit_hidden_dim=48,
        vit_inter_dim=96,
        vit_n_heads=3,
        vit_n_blocks=1,
        vit_img_size=32,
        vit_patch_size=16,
        vit_dropout=0.0,
        mp_pixel_shuffle_factor=2,
    )


def _hf_cfg(model_id: str, hidden_dim: int) -> VLMConfig:
    """
    Build a VLMConfig for the "hf" backend.

    Only three LM fields matter on the hf path: ``lm_backend`` (selects the wrapper),
    ``lm_model_type`` / ``lm_tokenizer`` (which HF model + tokenizer to load), and
    ``lm_hidden_dim`` (the ``Decoder`` asserts this equals the model's real hidden size).
    """
    cfg: VLMConfig = VLMConfig(
        lm_backend="hf",
        lm_model_type=model_id,
        lm_tokenizer=model_id,
        lm_hidden_dim=hidden_dim,
        **_tiny_vit_kwargs(),
    )
    return cfg


class TestRedesign(unittest.TestCase):
    # Shared, expensive fixtures built once for the whole class.
    cfg_smol: VLMConfig
    vlm_smol: VisionLanguageModel
    cfg_lfm: VLMConfig
    vlm_lfm: VisionLanguageModel

    @classmethod
    def setUpClass(cls) -> None:
        # A fixed seed makes the random weight init reproducible across runs.
        torch.manual_seed(0)

        # GROW regime: SmolLM2's embedding table equals its vocab, so adding the 66 VLM
        # tokens makes the tokenizer larger than the table -> resize must grow it.
        cls.cfg_smol = _hf_cfg(SMOL_ID, hidden_dim=960)
        cls.vlm_smol = VisionLanguageModel(cls.cfg_smol, load_backbone=False).eval()

        # SHRINK regime: LFM2.5's embedding table is padded well above its vocab, so even
        # after adding 66 tokens the tokenizer is still smaller than the table -> resize
        # must shrink it (and crucially does NOT init the new rows, which is why
        # _fit_embeddings_to_tokenizer is needed).
        cls.cfg_lfm = _hf_cfg(LFM_ID, hidden_dim=1024)
        cls.vlm_lfm = VisionLanguageModel(cls.cfg_lfm, load_backbone=False).eval()

    # Helper: the token ids of the 66 VLM special tokens, in config order.
    def _extra_ids(self, vlm: VisionLanguageModel) -> list[int]:
        extra_ids: list[int] = [
            vlm.tokenizer.convert_tokens_to_ids(tok_str)
            for tok_str in vlm.cfg.vlm_extra_tokens.values()
        ]
        return extra_ids

    # =====================================================================================
    # Group A — build_decoder dispatch (change #1: decoder construction was routed
    # through build_decoder, and the backend is chosen by cfg.lm_backend).
    # =====================================================================================

    def test_01_hf_backend_builds_decoder_wrapper(self) -> None:
        # On the "hf" backend the VLM's decoder must be the HF ``Decoder`` wrapper, not
        # the hand-written LanguageModel. This proves build_decoder is actually wired in.
        decoder: nn.Module = self.vlm_smol.decoder
        self.assertIsInstance(decoder, Decoder)

    def test_02_custom_backend_builds_language_model(self) -> None:
        # On the "custom" backend build_decoder must return the hand-written LanguageModel.
        # We call build_decoder directly with a tiny config so nothing is downloaded.
        custom_cfg: VLMConfig = VLMConfig(
            lm_backend="custom",
            lm_model_type="testing",
            lm_hidden_dim=64,
            lm_inter_dim=128,
            lm_vocab_size=100,
            lm_n_heads=4,
            lm_n_kv_heads=2,
            lm_n_blocks=2,
            lm_use_tokens=False,
            **_tiny_vit_kwargs(),
        )
        decoder: nn.Module = build_decoder(custom_cfg, load_backbone=False)
        self.assertIsInstance(decoder, LanguageModel)

    def test_03_unknown_backend_raises_value_error(self) -> None:
        # Any backend string other than "hf"/"custom" must fail loudly, not silently.
        bad_cfg: VLMConfig = VLMConfig(lm_backend="does-not-exist")
        with self.assertRaises(ValueError):
            build_decoder(bad_cfg, load_backbone=False)

    # =====================================================================================
    # Group B — tokenizer + special tokens (change #2: tokenizer built first, and it
    # carries the 66 VLM special tokens the embedding must accommodate).
    # =====================================================================================

    def test_04_tokenizer_registers_all_66_extra_tokens_uniquely(self) -> None:
        # The tokenizer must know every VLM special token, each mapped to a distinct id.
        extra_ids: list[int] = self._extra_ids(self.vlm_smol)
        unk_id: int = self.vlm_smol.tokenizer.unk_token_id if self.vlm_smol.tokenizer.unk_token_id is not None else -1

        # There are exactly 66 configured extra tokens.
        self.assertEqual(len(extra_ids), len(self.cfg_smol.vlm_extra_tokens))
        # None of them collapsed to the unknown-token id (i.e. all were really added).
        self.assertNotIn(unk_id, extra_ids)
        # All ids are unique -> no two special tokens share a row.
        self.assertEqual(len(set(extra_ids)), len(extra_ids))

    def test_05_image_token_id_is_exposed_and_consistent(self) -> None:
        # forward() relies on tokenizer.image_token_id to find image placeholders, so it
        # must exist and agree with converting the "<|image|>" string directly.
        image_token_id: int = self.vlm_smol.tokenizer.image_token_id
        image_token_str: str = self.cfg_smol.vlm_extra_tokens["image_token"]
        self.assertIsInstance(image_token_id, int)
        self.assertEqual(
            image_token_id,
            self.vlm_smol.tokenizer.convert_tokens_to_ids(image_token_str),
        )

    def test_06_tokenizer_attribute_is_set_on_the_vlm(self) -> None:
        # The tokenizer must be stored on the model (and be non-empty). Because the resize
        # step below already succeeded using it, this also implicitly confirms the
        # tokenizer was available *before* the decoder reconciliation ran.
        self.assertTrue(hasattr(self.vlm_smol, "tokenizer"))
        self.assertGreater(len(self.vlm_smol.tokenizer), 0)

    # =====================================================================================
    # Group C — embedding reconciliation, GROW regime (SmolLM2).
    # This is the core Blocker-2 fix (change #3).
    # =====================================================================================

    def test_07_input_embedding_rows_equal_tokenizer_length_grow(self) -> None:
        # After init, the input embedding table must have exactly one row per token id.
        n_tokens: int = len(self.vlm_smol.tokenizer)
        input_rows: int = self.vlm_smol.decoder.model.get_input_embeddings().weight.shape[0]
        self.assertEqual(input_rows, n_tokens)

    def test_08_output_head_rows_equal_tokenizer_length_grow(self) -> None:
        # The LM head (output embeddings) must also be sized to len(tokenizer); otherwise
        # the loss/logits vocab dimension would not match the token ids.
        n_tokens: int = len(self.vlm_smol.tokenizer)
        output_rows: int = self.vlm_smol.decoder.model.get_output_embeddings().weight.shape[0]
        self.assertEqual(output_rows, n_tokens)

    def test_09_all_special_token_ids_are_in_range_grow(self) -> None:
        # THE regression this whole change fixes: every special-token id (including the
        # image token) must index a valid embedding row. Before the resize, these ids were
        # >= the table size and an embedding lookup crashed.
        input_rows: int = self.vlm_smol.decoder.model.get_input_embeddings().weight.shape[0]
        extra_ids: list[int] = self._extra_ids(self.vlm_smol)
        self.assertLess(max(extra_ids), input_rows)
        self.assertLess(self.vlm_smol.tokenizer.image_token_id, input_rows)

    # =====================================================================================
    # Group D — embedding reconciliation, SHRINK regime (LFM2.5).
    # Same fix must hold even when the native table is *bigger* than the tokenizer.
    # =====================================================================================

    def test_10_resize_shrinks_padded_table_to_tokenizer_length(self) -> None:
        # Document + verify the shrink precondition: LFM's native table is larger than the
        # (already-extended) tokenizer, yet after init the table equals len(tokenizer).
        native_vocab: int = AutoConfig.from_pretrained(LFM_ID).vocab_size
        n_tokens: int = len(self.vlm_lfm.tokenizer)
        table_rows: int = self.vlm_lfm.decoder.model.get_input_embeddings().weight.shape[0]

        # Precondition: the native padded table really is bigger than what we need.
        self.assertGreater(native_vocab, n_tokens)
        # Result: resize brought the table down to exactly len(tokenizer).
        self.assertEqual(table_rows, n_tokens)

    def test_11_special_token_ids_in_range_after_shrink(self) -> None:
        # The out-of-range fix must also hold in the shrink regime.
        table_rows: int = self.vlm_lfm.decoder.model.get_input_embeddings().weight.shape[0]
        extra_ids: list[int] = self._extra_ids(self.vlm_lfm)
        self.assertLess(max(extra_ids), table_rows)
        self.assertLess(self.vlm_lfm.tokenizer.image_token_id, table_rows)

    def test_12_embedding_lookup_of_image_token_does_not_crash_shrink(self) -> None:
        # End-to-end proof: looking up the image-token id through the real embedding must
        # return a finite vector (this is exactly the operation that used to crash).
        image_token_id: int = self.vlm_lfm.tokenizer.image_token_id
        ids: torch.Tensor = torch.tensor([[image_token_id]], dtype=torch.long)  # [1, 1]
        embedded: torch.Tensor = self.vlm_lfm.decoder.token_embedding(ids)       # [1, 1, D]
        self.assertEqual(embedded.shape[-1], self.cfg_lfm.lm_hidden_dim)
        self.assertTrue(torch.isfinite(embedded).all())

    # =====================================================================================
    # Group E — _fit_embeddings_to_tokenizer behavior (change #3, the init half).
    # =====================================================================================

    def test_13_fit_initializes_extra_rows_to_small_normal(self) -> None:
        # After _fit, the special-token rows should look like N(0, 0.02): mean ~ 0 and a
        # standard deviation in a small band around 0.02.
        self.vlm_smol._fit_embeddings_to_tokenizer()  # re-run to be sure we measure its effect
        extra_ids: list[int] = self._extra_ids(self.vlm_smol)
        rows: torch.Tensor = self.vlm_smol.decoder.model.get_input_embeddings().weight[extra_ids]  # [66, D]

        mean: float = rows.mean().item()
        std: float = rows.std().item()
        self.assertLess(abs(mean), 0.01)          # centered near zero
        self.assertGreater(std, 0.01)             # not collapsed to a constant
        self.assertLess(std, 0.04)                # and not large -> consistent with 0.02

    def test_14_fit_does_not_touch_base_vocab_rows(self) -> None:
        # _fit must only rewrite the special-token rows. A plain base-vocab row (id 5)
        # must be identical before and after a _fit call.
        base_row_before: torch.Tensor = (
            self.vlm_smol.decoder.model.get_input_embeddings().weight[5].detach().clone()
        )
        self.vlm_smol._fit_embeddings_to_tokenizer()
        base_row_after: torch.Tensor = self.vlm_smol.decoder.model.get_input_embeddings().weight[5]
        self.assertTrue(torch.equal(base_row_before, base_row_after))

    def test_15_fit_exactly_number_of_changed_rows_equals_extra_tokens(self) -> None:
        # A stronger version of the previous test: snapshot the whole table, run _fit, and
        # confirm the set of rows that changed is exactly the 66 special-token ids.
        weight_before: torch.Tensor = (
            self.vlm_smol.decoder.model.get_input_embeddings().weight.detach().clone()
        )
        self.vlm_smol._fit_embeddings_to_tokenizer()
        weight_after: torch.Tensor = self.vlm_smol.decoder.model.get_input_embeddings().weight

        # A row "changed" if any element in it differs.
        changed_mask: torch.Tensor = (weight_before != weight_after).any(dim=1)  # [vocab]
        changed_ids: set[int] = set(torch.nonzero(changed_mask, as_tuple=True)[0].tolist())
        self.assertEqual(changed_ids, set(self._extra_ids(self.vlm_smol)))

    def test_16_fit_asserts_when_table_smaller_than_tokenizer(self) -> None:
        # _fit guards against being called before the resize: if the table has fewer rows
        # than the tokenizer, it must raise AssertionError. We build a tiny stub decoder
        # (10 rows) so we don't disturb the shared models.
        tiny_embedding: nn.Embedding = nn.Embedding(10, 8)
        stub_model: SimpleNamespace = SimpleNamespace(get_input_embeddings=lambda: tiny_embedding)
        stub_self: SimpleNamespace = SimpleNamespace(
            decoder=SimpleNamespace(model=stub_model),
            tokenizer=self.vlm_smol.tokenizer,   # len ~ 49k, far more than 10 rows
            cfg=self.cfg_smol,
        )
        with self.assertRaises(AssertionError):
            VisionLanguageModel._fit_embeddings_to_tokenizer(stub_self)

    # =====================================================================================
    # Group F — forward correctness (the changes must not break the training forward pass).
    # =====================================================================================

    def test_17_forward_without_targets_returns_hidden_states(self) -> None:
        # With no targets, forward returns raw decoder hidden states (head NOT applied) and
        # a None loss. The last dim must therefore be the hidden size, not the vocab size.
        input_ids: torch.Tensor = torch.randint(0, 100, (2, 6), dtype=torch.long)  # [B=2, T=6]
        with torch.no_grad():
            hidden: torch.Tensor
            loss: object
            hidden, loss = self.vlm_smol(input_ids, None)
        self.assertIsNone(loss)
        self.assertEqual(hidden.shape, (2, 6, self.cfg_smol.lm_hidden_dim))

    def test_18_forward_with_targets_returns_finite_scalar_loss(self) -> None:
        # With targets, forward applies the head and computes a cross-entropy loss, which
        # must be a finite scalar.
        input_ids: torch.Tensor = torch.randint(0, 100, (2, 6), dtype=torch.long)  # [B, T]
        targets: torch.Tensor = input_ids.clone()                                  # [B, T]
        with torch.no_grad():
            logits: torch.Tensor
            loss: torch.Tensor
            logits, loss = self.vlm_smol(input_ids, None, targets=targets)
        self.assertEqual(loss.ndim, 0)                       # scalar
        self.assertTrue(torch.isfinite(loss).all())
        # And the logits the loss is computed over span the full (resized) vocabulary.
        self.assertEqual(logits.shape[-1], len(self.vlm_smol.tokenizer))

    def test_19_head_vocab_dimension_matches_tokenizer_length(self) -> None:
        # The head that produces training logits must output exactly len(tokenizer) classes,
        # i.e. the resize propagated to the output side that the loss depends on.
        head: nn.Module = self.vlm_smol.decoder.head
        out_features: int = head.weight.shape[0]
        self.assertEqual(out_features, len(self.vlm_smol.tokenizer))

    def test_20_replace_img_tokens_only_swaps_image_positions(self) -> None:
        # _replace_img_tokens_with_embd must overwrite exactly the image-token positions
        # with the provided image embeddings (in order) and leave every other position
        # untouched. We fabricate embeddings directly so no ViT/MP forward is needed.
        image_token_id: int = self.vlm_smol.tokenizer.image_token_id
        dim: int = self.cfg_smol.lm_hidden_dim

        # Sequence: [text, IMG, text, IMG]  (2 image slots in one sample).
        input_ids: torch.Tensor = torch.tensor([[5, image_token_id, 7, image_token_id]], dtype=torch.long)  # [1, 4]
        token_embd: torch.Tensor = torch.zeros(1, 4, dim)          # text rows are all zeros
        image_embd: torch.Tensor = torch.arange(2 * dim, dtype=torch.float32).reshape(2, dim)  # [2, D], distinctive

        merged: torch.Tensor = self.vlm_smol._replace_img_tokens_with_embd(
            input_ids, token_embd, image_embd
        )  # [1, 4, D]

        # Image positions (index 1 and 3) got the two image rows, in order.
        self.assertTrue(torch.equal(merged[0, 1], image_embd[0]))
        self.assertTrue(torch.equal(merged[0, 3], image_embd[1]))
        # Text positions (index 0 and 2) are still the original zeros.
        self.assertTrue(torch.equal(merged[0, 0], torch.zeros(dim)))
        self.assertTrue(torch.equal(merged[0, 2], torch.zeros(dim)))

    # =====================================================================================
    # Group G — hidden-size propagation (Gap 4): the Decoder writes the HF model's real
    # hidden_size back into cfg.lm_hidden_dim so downstream components stay consistent.
    # =====================================================================================

    def test_21_cfg_hidden_dim_matches_model_hidden_size(self) -> None:
        # After construction, cfg.lm_hidden_dim must equal the loaded model's real
        # hidden_size for both models (960 for SmolLM2, 1024 for LFM2.5).
        self.assertEqual(
            self.vlm_smol.cfg.lm_hidden_dim,
            self.vlm_smol.decoder.model.config.hidden_size,
        )
        self.assertEqual(self.vlm_smol.cfg.lm_hidden_dim, 960)
        self.assertEqual(
            self.vlm_lfm.cfg.lm_hidden_dim,
            self.vlm_lfm.decoder.model.config.hidden_size,
        )
        self.assertEqual(self.vlm_lfm.cfg.lm_hidden_dim, 1024)

    def test_22_modality_projector_output_dim_follows_propagated_value(self) -> None:
        # The ModalityProjector must be built with the propagated hidden size, so its
        # projection output width matches both cfg and the model. This is the concrete
        # thing a wrong lm_hidden_dim would have silently broken.
        for vlm in (self.vlm_smol, self.vlm_lfm):
            mp_out: int = vlm.MP.proj.out_features
            model_hidden: int = vlm.decoder.model.config.hidden_size
            self.assertEqual(mp_out, vlm.cfg.lm_hidden_dim)
            self.assertEqual(mp_out, model_hidden)

    def test_24_vocab_propagated_from_resized_embedding(self) -> None:
        # Gap 4 (vocab): cfg.lm_vocab_size must equal the ACTUAL resized embedding row count
        # (== len(tokenizer)), not the SmolLM default, and base + extra must reconstruct it.
        for vlm, cfg in ((self.vlm_smol, self.cfg_smol), (self.vlm_lfm, self.cfg_lfm)):
            n_tokens: int = len(vlm.tokenizer)
            rows: int = vlm.decoder.model.get_input_embeddings().weight.shape[0]
            self.assertEqual(rows, n_tokens)                       # resize really happened
            self.assertEqual(cfg.lm_vocab_size, rows)              # cfg tracks the real vocab
            self.assertEqual(
                cfg.lm_base_vocab_size + cfg.extra_token_amount, cfg.lm_vocab_size
            )

    def test_25_lfm_vocab_is_not_the_padded_native_value(self) -> None:
        # The whole reason vocab is set post-resize: LFM's native config.vocab_size is padded
        # (65536) and must NOT be what cfg records; cfg must follow the tokenizer instead.
        native_vocab: int = AutoConfig.from_pretrained(LFM_ID).vocab_size
        self.assertNotEqual(self.cfg_lfm.lm_vocab_size, native_vocab)
        self.assertEqual(self.cfg_lfm.lm_vocab_size, len(self.vlm_lfm.tokenizer))

    def test_26_arch_fields_propagated_from_hf_config(self) -> None:
        # Gap 4 (arch): the Decoder must copy the HF model's real architecture into cfg so
        # the saved config is honest rather than carrying SmolLM defaults. Check against each
        # model's actual HF config values (SmolLM: 15/5/32 heads/kv/layers; LFM2: 16/8/14).
        for vlm, cfg in ((self.vlm_smol, self.cfg_smol), (self.vlm_lfm, self.cfg_lfm)):
            hf = vlm.decoder.model.config
            self.assertEqual(cfg.lm_inter_dim, hf.intermediate_size)
            self.assertEqual(cfg.lm_n_heads, hf.num_attention_heads)
            self.assertEqual(cfg.lm_n_kv_heads, hf.num_key_value_heads)
            self.assertEqual(cfg.lm_n_blocks, hf.num_hidden_layers)
            self.assertEqual(cfg.lm_max_position_embeddings, hf.max_position_embeddings)

    def test_27_tie_weights_flag_is_honored_both_ways(self) -> None:
        # Gap 5 (tie weights): cfg.lm_tie_weights must actually take effect on the hf path.
        # data_ptr() equality is ground truth for "input embedding and head are the SAME
        # matrix". Build a fresh VLM per (model, flag) so we don't disturb the shared ones.
        for model_id, hidden in ((SMOL_ID, 960), (LFM_ID, 1024)):
            for tie in (True, False):
                with self.subTest(model=model_id, tie=tie):
                    cfg: VLMConfig = _hf_cfg(model_id, hidden_dim=hidden)
                    cfg.lm_tie_weights = tie
                    vlm: VisionLanguageModel = VisionLanguageModel(cfg, load_backbone=False).eval()

                    ie: torch.Tensor = vlm.decoder.model.get_input_embeddings().weight
                    oe: torch.Tensor = vlm.decoder.model.get_output_embeddings().weight
                    shared: bool = ie.data_ptr() == oe.data_ptr()

                    self.assertEqual(shared, tie)                          # flag actually applied
                    self.assertEqual(vlm.decoder.model.config.tie_word_embeddings, tie)  # config in sync
                    # Both matrices span the full resized vocab either way.
                    self.assertEqual(ie.shape[0], len(vlm.tokenizer))
                    self.assertEqual(oe.shape[0], len(vlm.tokenizer))

    def test_28_untied_model_still_forwards_and_head_is_independent(self) -> None:
        # Gap 5 (tie weights), regression guard: untying replaces the head with a fresh
        # Parameter, so prove that (a) a full forward+loss still works and produces logits
        # over the resized vocab, and (b) the head and embedding are now truly independent
        # — editing one must not change the other.
        cfg: VLMConfig = _hf_cfg(SMOL_ID, hidden_dim=960)
        cfg.lm_tie_weights = False
        vlm: VisionLanguageModel = VisionLanguageModel(cfg, load_backbone=False).eval()

        # (a) forward with targets still yields a finite scalar loss over the full vocab.
        input_ids: torch.Tensor = torch.randint(0, 100, (2, 6), dtype=torch.long)  # [B, T]
        targets: torch.Tensor = input_ids.clone()
        with torch.no_grad():
            logits: torch.Tensor
            loss: torch.Tensor
            logits, loss = vlm(input_ids, None, targets=targets)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss).all())
        self.assertEqual(logits.shape[-1], len(vlm.tokenizer))

        # (b) independence: mutate the head row 0, embedding row 0 must be unaffected.
        ie: torch.Tensor = vlm.decoder.model.get_input_embeddings().weight
        oe: torch.Tensor = vlm.decoder.model.get_output_embeddings().weight
        embed_row_before: torch.Tensor = ie[0].detach().clone()
        with torch.no_grad():
            oe[0] += 1.0
        self.assertTrue(torch.equal(ie[0], embed_row_before))  # embedding untouched by head edit

    def test_29_head_of_base_matches_full_model_logits(self) -> None:
        # Gap 5 (final-norm assumption): the VLM produces logits in two manual steps,
        # head(base(x)) (vision_language_model.py: logits = self.decoder.head(self.decoder(...))).
        # That is only correct if base == get_decoder() already applies the model's FINAL
        # normalization before returning last_hidden_state. Prove it by checking head(base(x))
        # equals the model's OWN .logits. If a future backbone returned pre-final-norm hidden
        # states, this fails loudly instead of silently degrading training.
        #
        # Fresh, float32 models: a shared bf16 LFM would hit the (separate) mixed-dtype issue;
        # .float() isolates this check to the norm/head wiring.
        for model_id, hidden in ((SMOL_ID, 960), (LFM_ID, 1024)):
            with self.subTest(model=model_id):
                cfg: VLMConfig = _hf_cfg(model_id, hidden_dim=hidden)
                vlm: VisionLanguageModel = VisionLanguageModel(cfg, load_backbone=False).eval().float()
                decoder = vlm.decoder

                input_ids: torch.Tensor = torch.randint(0, 100, (2, 5), dtype=torch.long)  # [B, T]
                x: torch.Tensor = decoder.token_embedding(input_ids)                        # [B, T, D]
                with torch.no_grad():
                    # Manual path: exactly what the VLM does — body then head, separately.
                    manual_logits: torch.Tensor = decoder.head(decoder.base(inputs_embeds=x).last_hidden_state)
                    # Reference path: the full CausalLM's own logits over the same input.
                    official_logits: torch.Tensor = decoder.model(inputs_embeds=x).logits

                self.assertEqual(manual_logits.shape, official_logits.shape)
                self.assertEqual(manual_logits.shape, (2, 5, len(vlm.tokenizer)))
                # If base skipped the final norm, these would diverge well beyond fp rounding.
                self.assertTrue(torch.allclose(manual_logits, official_logits, atol=1e-4, rtol=1e-4))

    def test_30_decoder_loads_in_fp32_and_vlm_is_uniform(self) -> None:
        # Dtype consistency: HF backbones ship in bf16 (config.torch_dtype), but train.py uses
        # autocast(bf16), which needs fp32 MASTER weights. The Decoder must force fp32 so the
        # LM matches the fp32 ViT/ModalityProjector and the custom-LM path. Assert every param
        # across decoder + ViT + MP is float32 (no stray bf16 slipping through).
        for vlm in (self.vlm_smol, self.vlm_lfm):
            decoder_dtypes: set = {p.dtype for p in vlm.decoder.model.parameters()}
            self.assertEqual(decoder_dtypes, {torch.float32})
            vit_dtypes: set = {p.dtype for p in vlm.vision_encoder.parameters()}
            mp_dtypes: set = {p.dtype for p in vlm.MP.parameters()}
            self.assertEqual(vit_dtypes, {torch.float32})
            self.assertEqual(mp_dtypes, {torch.float32})
            # The whole model is one consistent precision.
            self.assertEqual({p.dtype for p in vlm.parameters()}, {torch.float32})

    def test_23_wrong_cfg_hidden_dim_warns_and_is_corrected(self) -> None:
        # If the incoming cfg.lm_hidden_dim disagrees with the HF model, the Decoder must
        # emit a warning (not raise) and overwrite cfg with the model's real value. Here we
        # feed LFM2.5 (hidden 1024) a deliberately wrong 999 and check it self-corrects.
        wrong_cfg: VLMConfig = _hf_cfg(LFM_ID, hidden_dim=999)
        with self.assertWarns(UserWarning):
            decoder: Decoder = Decoder(wrong_cfg, load_backbone=False)
        self.assertEqual(wrong_cfg.lm_hidden_dim, decoder.model.config.hidden_size)
        self.assertEqual(wrong_cfg.lm_hidden_dim, 1024)


if __name__ == "__main__":
    unittest.main()

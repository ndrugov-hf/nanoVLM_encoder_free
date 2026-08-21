"""Rigorous tests for the packing carry-through (sub-step 3).

`ConstantLengthDataset` packs several finished samples end-to-end into one fixed-length training
sequence. On the encoder-free path a sample's "images" is a LIST of per-image dicts
({"pixel_values", "image_position_ids"}). The packing code treats each image entry opaquely --
`len(sample["images"])` to count, `ims.extend(sample["images"])` to concatenate -- so per-image
dicts should pass through unchanged. These tests CONFIRM that (they are expected to pass as-is; a
failure means the per-image list does NOT survive packing and packing needs a fix).

Two seams are exercised directly, no model / tokenizer / GPU:
  * `_pack_one_group` -- the glue: concatenates the grouped samples' image lists, in group order.
  * `_balanced_greedy_knapsack` -- the counting: uses len(images) and the per-group image limit.
Plus one end-to-end pass through the whole `ConstantLengthDataset`.
"""

import unittest

import torch

from data.advanced_datasets import ConstantLengthDataset


# --- fakes / builders -------------------------------------------------------------------------

class _FakeInner:
    """`ConstantLengthDataset.__init__` only reads `dataset.mp_image_token_length` (for the unused
    __len__ estimate). The two packing helpers under test never touch the inner dataset at all."""

    mp_image_token_length = 0


def _img(tag: int, k: int) -> dict:
    """One identifiable per-image dict in the encoder-free shape, tagged (tag, k) so it can be
    tracked by identity through packing. Packing is shape-agnostic, so tiny tensors suffice."""
    return {
        "pixel_values": torch.tensor([float(tag * 100 + k)]),
        "image_position_ids": torch.tensor([tag, k], dtype=torch.long),
    }


def _sample(n_images: int, seq_len: int, tag: int) -> dict:
    """A finished sample as `ConstantLengthDataset` sees it: 1D text tensors + a list of per-image
    dicts (empty list for a text-only sample)."""
    ids = torch.arange(seq_len)
    return {
        "input_ids": ids,
        "labels": ids.clone(),
        "attention_mask": torch.ones(seq_len, dtype=torch.long),
        "images": [_img(tag, k) for k in range(n_images)],
    }


def _cld(**kw) -> ConstantLengthDataset:
    return ConstantLengthDataset(_FakeInner(), **kw)


# --- the glue: _pack_one_group ----------------------------------------------------------------

class TestPackOneGroupGlue(unittest.TestCase):
    """`_pack_one_group` concatenates the grouped samples' per-image lists into one flat list, in
    group order, leaving the dicts themselves untouched."""

    def setUp(self):
        self.cld = _cld(seq_length=1000)

    def test_concatenates_image_lists_in_group_order(self):
        # sample 0: 2 images; sample 1: text-only (contributes nothing); sample 2: 1 image.
        buffer = [_sample(2, 3, tag=0), _sample(0, 3, tag=1), _sample(1, 3, tag=2)]
        _, _, _, ims = self.cld._pack_one_group([0, 1, 2], buffer, max_len=1000)
        expected = buffer[0]["images"] + buffer[1]["images"] + buffer[2]["images"]
        self.assertEqual(len(ims), 3)
        # SAME dict objects, in order -- nothing copied, dropped, reordered, or mangled.
        for got, exp in zip(ims, expected):
            self.assertIs(got, exp)

    def test_follows_group_indices_order_not_buffer_order(self):
        # group order [2, 0] must place sample 2's images first, then sample 0's -- the images'
        # order has to track the order the samples are concatenated into the sequence.
        buffer = [_sample(2, 3, tag=0), _sample(0, 3, tag=1), _sample(1, 3, tag=2)]
        _, _, _, ims = self.cld._pack_one_group([2, 0], buffer, max_len=1000)
        expected = buffer[2]["images"] + buffer[0]["images"]
        self.assertEqual([id(x) for x in ims], [id(x) for x in expected])

    def test_text_only_group_yields_empty_image_list(self):
        buffer = [_sample(0, 3, tag=0), _sample(0, 4, tag=1)]
        _, _, _, ims = self.cld._pack_one_group([0, 1], buffer, max_len=1000)
        self.assertEqual(ims, [])

    def test_single_sample_group(self):
        buffer = [_sample(3, 5, tag=7)]
        _, _, _, ims = self.cld._pack_one_group([0], buffer, max_len=1000)
        self.assertEqual([id(x) for x in ims], [id(x) for x in buffer[0]["images"]])
        self.assertEqual(len(ims), 3)

    def test_text_is_concatenated_alongside_images(self):
        # The glue keeps text and images consistent: packed input_ids is the grouped samples'
        # input_ids concatenated, and the image list is their images concatenated.
        buffer = [_sample(1, 3, tag=0), _sample(2, 4, tag=1)]
        ids, _, _, ims = self.cld._pack_one_group([0, 1], buffer, max_len=1000)
        self.assertEqual(tuple(ids.shape), (7,))  # 3 + 4 tokens
        self.assertTrue(torch.equal(ids, torch.cat([buffer[0]["input_ids"], buffer[1]["input_ids"]])))
        self.assertEqual(len(ims), 3)             # 1 + 2 images

    def test_image_dicts_are_not_mutated(self):
        buffer = [_sample(2, 3, tag=0)]
        before = [(d["pixel_values"].clone(), d["image_position_ids"].clone())
                  for d in buffer[0]["images"]]
        self.cld._pack_one_group([0], buffer, max_len=1000)
        for d, (pv, pos) in zip(buffer[0]["images"], before):
            self.assertTrue(torch.equal(d["pixel_values"], pv))
            self.assertTrue(torch.equal(d["image_position_ids"], pos))

    def test_over_length_group_raises(self):
        # Safety guard (line 234): packing more tokens than max_len must fail loudly.
        buffer = [_sample(1, 6, tag=0), _sample(1, 6, tag=1)]  # 12 tokens
        with self.assertRaises(ValueError):
            self.cld._pack_one_group([0, 1], buffer, max_len=8)


# --- the counting: _balanced_greedy_knapsack --------------------------------------------------

class TestKnapsackImageCounting(unittest.TestCase):
    """`_balanced_greedy_knapsack` counts a sample's images with len(sample["images"]) and keeps
    each packed group's total image count within `max_images_per_knapsack`."""

    def setUp(self):
        self.cld = _cld(seq_length=1000)

    @staticmethod
    def _placed(groups) -> list[int]:
        return sorted(i for g in groups for i in g)

    def test_every_sample_placed_exactly_once(self):
        buffer = [_sample(n, 2, tag=t) for t, n in enumerate([3, 3, 3, 3])]
        groups = self.cld._balanced_greedy_knapsack(buffer, 1000, delta=0, max_images_per_knapsack=6)
        self.assertEqual(self._placed(groups), [0, 1, 2, 3])  # all present, no dup, no drop

    def test_no_group_exceeds_the_image_limit(self):
        # 4 samples x 3 images, limit 6 -> at most 2 samples per group. If counting used anything
        # other than len(images) (e.g. counted dict keys), a group would blow past the limit.
        buffer = [_sample(3, 2, tag=t) for t in range(4)]
        limit = 6
        groups = self.cld._balanced_greedy_knapsack(buffer, 1000, delta=0, max_images_per_knapsack=limit)
        for g in groups:
            self.assertLessEqual(sum(len(buffer[i]["images"]) for i in g), limit)
        self.assertEqual(self._placed(groups), [0, 1, 2, 3])

    def test_length_constraint_still_respected(self):
        # Both constraints active: length limits to 2 samples/group, image limit is slack.
        buffer = [_sample(1, 4, tag=t) for t in range(4)]  # 4 tokens, 1 image each
        L = 8
        groups = self.cld._balanced_greedy_knapsack(buffer, L, delta=0, max_images_per_knapsack=100)
        for g in groups:
            self.assertLessEqual(sum(len(buffer[i]["input_ids"]) for i in g), L)
        self.assertEqual(self._placed(groups), [0, 1, 2, 3])

    def test_no_image_limit_places_all(self):
        # max_images_per_knapsack=None -> image count never blocks; all samples still placed once.
        buffer = [_sample(5, 2, tag=t) for t in range(3)]
        groups = self.cld._balanced_greedy_knapsack(buffer, 1000, delta=0, max_images_per_knapsack=None)
        self.assertEqual(self._placed(groups), [0, 1, 2])

    def test_text_only_samples_count_as_zero_images(self):
        # Even with image limit 1, five text-only samples (images == []) all pack together, because
        # len([]) == 0. If empty lists miscounted, the limit would wrongly split them.
        buffer = [_sample(0, 2, tag=t) for t in range(5)]
        groups = self.cld._balanced_greedy_knapsack(buffer, 1000, delta=0, max_images_per_knapsack=1)
        self.assertEqual(self._placed(groups), [0, 1, 2, 3, 4])


# --- end to end through the whole dataset -----------------------------------------------------

class _FakeVQADataset:
    """Minimal stand-in for the inner VQADataset that the producer thread drives: it needs
    mp_image_token_length, a `.dataset` with __len__ (selects the index-based iterator), a
    `.tokenizer.pad_token_id` (the producer appends a pad token), and __len__/__getitem__."""

    mp_image_token_length = 0

    def __init__(self, samples):
        self._samples = samples
        self.dataset = list(range(len(samples)))  # has __len__ -> index path in make_base_iterator
        self.tokenizer = type("_Tok", (), {"pad_token_id": 0})()

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, i):
        return self._samples[i]


class TestConstantLengthDatasetEndToEnd(unittest.TestCase):
    def test_full_pass_preserves_every_per_image_dict(self):
        # Run the real packing pipeline (producer thread + knapsack + pack) over encoder-free
        # samples and confirm every original per-image dict comes out the other side, intact.
        samples = [_sample(2, 5, tag=0), _sample(0, 5, tag=1), _sample(1, 5, tag=2)]
        cld = ConstantLengthDataset(
            _FakeVQADataset(samples), infinite=False, max_sample_length=1000,
            seq_length=1000, num_of_sequences=1, max_images_per_example=10,
            max_images_per_knapsack=100,
        )
        packed = list(cld)

        all_imgs = [d for p in packed for d in p["images"]]
        self.assertEqual(len(all_imgs), 3)  # 2 + 0 + 1 images, none lost
        for d in all_imgs:
            self.assertIn("pixel_values", d)
            self.assertIn("image_position_ids", d)
        # every packed sample's images is a list (the shape the collator expects), and its text is
        # a stacked 1D tensor -- text and images travelled together.
        for p in packed:
            self.assertIsInstance(p["images"], list)
            self.assertEqual(p["input_ids"].dim(), 1)


if __name__ == "__main__":
    unittest.main()

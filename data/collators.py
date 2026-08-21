import torch

from torch import Tensor
from jaxtyping import Int, Float

class BaseCollator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def _pad_batch(self, batch, max_length):
        batch["input_ids"] = [torch.nn.functional.pad(ids, (max_length - len(ids), 0), value=self.tokenizer.pad_token_id) for ids in batch["input_ids"]]
        batch["labels"]    = [torch.nn.functional.pad(labels, (max_length - len(labels), 0), value=self.tokenizer.pad_token_id) for labels in batch["labels"]]
        batch["attention_mask"] = [torch.nn.functional.pad(attention_mask, (max_length - len(attention_mask), 0), value=0) for attention_mask in batch["attention_mask"]]

    def prepare_batch(self, batch, max_length=None):
        # 1) Handle empty
        if not batch:
            return {"input_ids": [], "labels": [], "attention_mask": [], "images": []}

        # 2) Drop None rows
        batch = [s for s in batch if s is not None]
        if not batch:
            return {"input_ids": [], "labels": [], "attention_mask": [], "images": []}

        # batch is a list of dicts, each containing "input_ids", "attention_mask", "labels", "images"
        # let's convert it to a dict of lists of tensors
        batch = {k: [item[k] for item in batch] for k in batch[0]}

        if max_length is not None:
            batch = self._discard_samples_that_are_too_long(batch, max_length)

        if len(batch["input_ids"]) == 0:
            return batch

        # Pad samples to max length
        if max_length is not None:
            max_len = max_length
        else:
            max_len = max(map(len, batch["input_ids"]))
        self._pad_batch(batch, max_len) #  dictionaries in Python are mutable and passed by reference

        return {
            "input_ids": torch.stack(batch["input_ids"]),
            "attention_mask": torch.stack(batch["attention_mask"]),
            "images": batch["images"],
            "labels": torch.stack(batch["labels"]),
        }

    # ------------------------ CHANGE ------------------------
    # New function:
    def prepare_batch_encoder_free(self, batch, max_length=None):
        # 1) Handle empty
        if not batch:
            return {"input_ids": [], "labels": [], "attention_mask": [], "images": None}
        
        # 2) Drop None rows
        batch = [s for s in batch if s is not None]
        if not batch:
            return {"input_ids": [], "labels": [], "attention_mask": [], "images": None}

        # batch is a list of dicts, each containing "input_ids", "attention_mask", "labels", "images"
        # let's convert it to a dict of lists of tensors
        batch = {k: [item[k] for item in batch] for k in batch[0]}

        # Discard samples that are too long 
        if max_length is not None:
            batch = self._discard_samples_that_are_too_long(batch, max_length)

        if len(batch["input_ids"]) == 0:
            return {"input_ids": [], "labels": [], "attention_mask": [], "images": None}

        if sum(len(per_sample_list) for per_sample_list in batch["images"]) == 0:
            # Set batch['images'] to None if there are no images in the batch
            batch["images"] = None
        else:
            # Otherwise, batch['images'] is a list of lists of dicts - one list per sample. Turn it into a dict of tensors:
            # {"pixel_values": (num_images, patches, model_flat_patch_dim), "image_position_ids": (num_images, patches, 2)}
            pixel_values_list = []
            image_position_ids_list = []

            for per_sample_list in batch["images"]:
                for per_image_dict in per_sample_list:
                    pixel_values: Float[Tensor, "patches model_flat_patch_dim"] = per_image_dict["pixel_values"]
                    pixel_values_list.append(pixel_values)

                    image_position_ids: Int[Tensor, "patches 2"] = per_image_dict["image_position_ids"]
                    image_position_ids_list.append(image_position_ids)

            pixel_values_tensor: Float[Tensor, "num_images patches model_flat_patch_dim"] = torch.stack(pixel_values_list, dim=0)
            image_position_ids_tensor: Int[Tensor, "num_images patches 2"] = torch.stack(image_position_ids_list, dim=0)
            batch["images"] = {"pixel_values": pixel_values_tensor, "image_position_ids": image_position_ids_tensor}

        # Pad samples to max length
        if max_length is not None:
            max_len = max_length
        else:
            max_len = max(map(len, batch["input_ids"]))
        self._pad_batch(batch, max_len) #  dictionaries in Python are mutable and passed by reference

        return {
            "input_ids": torch.stack(batch["input_ids"]),
            "attention_mask": torch.stack(batch["attention_mask"]),
            "images": batch["images"],
            "labels": torch.stack(batch["labels"]),
        }
    # --------------------- END OF CHANGE ---------------------
        
    def _discard_samples_that_are_too_long(self, batch, max_length):
        filtered = [
            (ids, label, attn, img)
            for ids, label, attn, img in zip(batch["input_ids"], batch["labels"], batch["attention_mask"], batch["images"])
            if len(ids) <= max_length
        ]
        if not filtered:
            return {"input_ids": [], "labels": [], "attention_mask": [], "images": []}
        batch_token_ids, batch_labels, batch_attentions, batch_images = zip(*filtered)
        return {"input_ids": list(batch_token_ids), "labels": list(batch_labels), "attention_mask": list(batch_attentions), "images": list(batch_images)}


class VQACollator(BaseCollator):  # Visual Question Answering Collator
    def __init__(self, tokenizer, max_length, vision_backend="vit"):
        self.max_length = max_length
        self.vision_backend = vision_backend
        super().__init__(tokenizer)

    def _pad_batch(self, batch, max_length):  # Reimplementing to use -100 as the pad value for labels, so that it's ignored by the loss
        batch["input_ids"] = [torch.nn.functional.pad(ids, (max_length - len(ids), 0), value=self.tokenizer.pad_token_id) for ids in batch["input_ids"]]
        batch["labels"]    = [torch.nn.functional.pad(labels, (max_length - len(labels), 0), value=-100) for labels in batch["labels"]]
        batch["attention_mask"] = [torch.nn.functional.pad(attention_mask, (max_length - len(attention_mask), 0), value=0) for attention_mask in batch["attention_mask"]]

    def __call__(self, batch):
        # ------------------------ CHANGE ------------------------
        # Added the if-else branching. Before, there was no branching, and the vision_backend="vit"
        # branch was always executed
        if self.vision_backend == "encoder_free":
            batch = self.prepare_batch_encoder_free(batch, max_length=self.max_length)
        else:
            batch = self.prepare_batch(batch, max_length=self.max_length)
        # --------------------- END OF CHANGE ---------------------
        return batch
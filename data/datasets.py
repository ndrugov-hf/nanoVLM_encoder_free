import torch
from PIL import Image
from torch.utils.data import Dataset
from data.processors import get_image_string, get_image_string_encoder_free
import logging

from jaxtyping import Int, Float
from torch import Tensor

class BaseDataset(Dataset):
    def __init__(self, dataset, tokenizer, image_processor, mp_image_token_length, relevance_min_rating=1, image_correspondence_min_rating=1, visual_dependency_min_rating=1, formatting_min_rating=1, vision_backend="vit"):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.mp_image_token_length = mp_image_token_length
        self.relevance_min_rating = relevance_min_rating
        self.image_correspondence_min_rating = image_correspondence_min_rating
        self.visual_dependency_min_rating = visual_dependency_min_rating
        self.formatting_min_rating = formatting_min_rating
        self.vision_backend = vision_backend
        self.prefix_len = self._get_prefix_len()

    def __len__(self):
        return len(self.dataset)

    def _get_prefix_len(self):
        random_string_5_letters = "xzyvd"
        random_string_chat_templated = self.tokenizer.apply_chat_template([{"role": "assistant", "content": random_string_5_letters}], tokenize=False, add_special_tokens=False)
        random_string_location = random_string_chat_templated.find(random_string_5_letters)
        return len(self.tokenizer.encode(random_string_chat_templated[:random_string_location]))

    def _get_messages(self, item, image_string):
        messages = []
        for index, text in enumerate(item['texts']):
            try:
                if item.get('relevance_ratings') is not None and item['relevance_ratings'][index] is not None and item['relevance_ratings'][index] < self.relevance_min_rating:
                    continue
                if item.get('image_correspondence_ratings') is not None and item['image_correspondence_ratings'][index] is not None and item['image_correspondence_ratings'][index] < self.image_correspondence_min_rating:
                    continue
                if item.get('visual_dependency_ratings') is not None and item['visual_dependency_ratings'][index] is not None and item['visual_dependency_ratings'][index] < self.visual_dependency_min_rating:
                    continue
                if item.get('formatting_ratings') is not None and item['formatting_ratings'][index] is not None and item['formatting_ratings'][index] < self.formatting_min_rating:
                    continue
            except Exception as e:
                logging.warning(f"Error processing item: {item}, index: {index}: {e}")

            messages.append({"role": "user", "content": text['user']})
            messages.append({"role": "assistant", "content": text['assistant']})

        if len(messages) == 0:
            return messages

        # Safety check to ensure no image tokens are present in the text before adding them.
        for msg in messages:
            if self.tokenizer.image_token in msg["content"]:
                logging.warning(f"Found and removed an image token in the {msg['role']} text before adding the image string.")
                msg["content"] = msg["content"].replace(self.tokenizer.image_token, "")

        # ------------------------ CHANGE ------------------------
        # OLD:
        #
        # if len(splitted_image_counts) > 0:
        #     image_string = get_image_string(self.tokenizer, splitted_image_counts, self.mp_image_token_length)
        #     messages[0]["content"] = image_string + messages[0]["content"]
        #
        # NEW:
        messages[0]["content"] = image_string + messages[0]["content"]
        # --------------------- END OF CHANGE ---------------------

        return messages

    def _process_images(self, images):
        processed_images = []
        splitted_image_counts = []
        for image in images:
            if isinstance(image, Image.Image):
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                processed_image, splitted_image_count = self.image_processor(image)
                if not hasattr(self.tokenizer, "global_image_token") and splitted_image_count[0]*splitted_image_count[1] == len(processed_image) - 1:
                    # If the tokenizer doesn't have a global image token, but the processor generated it, remove it
                    processed_image = processed_image[1:]
                processed_images.append(processed_image)
                splitted_image_counts.append(splitted_image_count)
            else:
                raise ValueError(f"Error processing image: {image}")
        return processed_images, splitted_image_counts
    
    def _process_images_encoder_free(self, images):
        """
        Purpose:
            Turn one sample's raw images into the encoder-free image data, kept as one entry per
            image. Each image is run through the ImageProcessor on its own (resize -> cut into
            patches -> merge each k x k block of patches into one model patch -> pad to a fixed
            length), and its result becomes its own dict in a list. The processor is called per
            image, so each call is a batch of one; the leading batch axis is dropped so each entry
            holds that single image's data.

            The per-image list (rather than one stacked array for the whole sample) is what lets the
            packing step count a sample's images with `len(...)` and glue several samples' images
            together; the collator stacks the whole batch's images into one array later.

            Encoder-free sibling of `_process_images` (the ViT path), which is left untouched.

        Parameters:
         * images : the sample's raw images, a list of PIL images (or (C, H, W) tensors). Each is
                    processed on its own, since different aspect ratios resize to different sizes.

        Returns:
            A tuple (processed_images, num_soft_tokens_per_image):
             * processed_images : list with one dict per image, in image order. Each dict is
                   {"pixel_values":       (max_soft_tokens, model_flat_patch_dim) float,
                    "image_position_ids": (max_soft_tokens, 2) int, where (-1, -1) marks a
                                          padding patch}.
             * num_soft_tokens_per_image : list[int], one entry per image, the real (non-padding)
                   patch count for that image. Equals the number of real rows in that image's
                   pixel_values, so the text can write exactly that many <|image|> placeholders.
        """
        processed_images = []
        num_soft_tokens_per_image = []

        for image in images:
            processed_image_data = self.image_processor(image)
        
            processed_images.append({"pixel_values": processed_image_data["pixel_values"].squeeze(0), 
                                     "image_position_ids":  processed_image_data["image_position_ids"].squeeze(0)})
            
            num_soft_tokens_per_image.append(processed_image_data["num_soft_tokens_per_image"][0])

        return processed_images, num_soft_tokens_per_image

    def _prepare_inputs_and_loss_mask(self, messages):
        conv_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_special_tokens=False,
            return_dict=True,
        )
        mask = [0] * len(conv_ids["input_ids"])

        # Locate each assistant turn and flip its mask to 1
        cursor = 0
        for msg in messages:
            # -------------- CHANGE ----------
            # Old:
            #   segment_ids = self.tokenizer.apply_chat_template(
            #       [msg], tokenize=True, add_special_tokens=False
            #   )
            #   seg_len = len(segment_ids)
            #   transformers 5 makes apply_chat_template(tokenize=True) return a
            #   BatchEncoding instead of a list, so len(segment_ids) was 2 (the key
            #   count) instead of the token count -> empty loss mask -> nan loss.
            # New:
            segment_ids = self.tokenizer.apply_chat_template(
                [msg], tokenize=True, add_special_tokens=False, return_dict=True
            )
            seg_len = len(segment_ids["input_ids"])
            # -------------- END OF CHANGE ----------

            if msg["role"] == "assistant":
                start = cursor + self.prefix_len
                end   = cursor + seg_len
                mask[start:end] = [1] * (end - start)  # attend to these tokens

            cursor += seg_len
        
        return torch.tensor(conv_ids["input_ids"]), torch.tensor(mask).to(torch.bool), torch.tensor(conv_ids["attention_mask"])


class VQADataset(BaseDataset):  # Visual Question Answering Dataset
    def iter_for_worker(self):  # with iterable datasets, each worker gets different shards
        for data in self.dataset:
            yield self._process_data(data)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return self._process_data(item)

    def _process_data(self, item):
        # Handle images (should be a list)
        if item['images'] is None:
            images_data = []
        else:
            images_data = item['images']
            if not isinstance(images_data, list):
                images_data = [images_data]

        # ------------------------ CHANGE ------------------------
        # Added the if-else branching. Before, there was no branching, and the vision_backend="vit"
        # branch was always executed
        image_string = ""
        processed_images = []

        if self.vision_backend == "encoder_free":
            processed_images, num_soft_tokens_per_image = self._process_images_encoder_free(images_data)
            image_string: str = get_image_string_encoder_free(self.tokenizer, num_soft_tokens_per_image)

        else:
            splitted_image_counts = []
            if images_data: # Only process if there are images
                processed_images, splitted_image_counts = self._process_images(images_data)

            if len(splitted_image_counts) > 0:
                image_string = get_image_string(self.tokenizer, splitted_image_counts, self.mp_image_token_length)

        # --------------------- END OF CHANGE ---------------------

        messages = self._get_messages(item, image_string)

        if len(messages) == 0:
            return None

        input_ids, mask, attention_mask = self._prepare_inputs_and_loss_mask(messages)
        labels = self._get_labels(input_ids, mask)

        return {
            "images": processed_images,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _get_labels(self, input_ids, mask):
        labels = input_ids.clone().masked_fill(~mask, -100)
        labels = labels.roll(-1) # Shift labels for causal LM
        labels[-1] = -100 # Last token has no target
        
        return labels
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration, AutoProcessor
from PIL import Image
import torch
import os

from geolingual.config import DESCRIPTIONS_DIR, VLM_DIR_NAME

class Model:
    def __init__(self, model_id: str):
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model_id = model_id
        if "A3B" in model_id:
            self.model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                model_id,
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            )
        else:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_id,
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            )
        with open("prompts/satellite_prompt.txt", "r") as f:
            self.satellite_prompt = f.read()
        with open("prompts/panorama_prompt.txt", "r") as f:
            self.street_prompt = f.read()

    def load_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        original_width, original_height = image.size
        new_size = (original_width // 2, original_height // 2)
        image = image.resize(new_size, Image.Resampling.LANCZOS)
        return image


    def message_template(self, image_path, image_type = "satellite"):

        image = self.load_image(image_path)
        if image_type == "satellite":
            prompt = self.satellite_prompt
        else:
            prompt = self.street_prompt

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return messages
    def generate(self, image_path, image_type="satellite"):
        messages = self.message_template(image_path, image_type)

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        out_dir = os.path.join(DESCRIPTIONS_DIR, image_type, VLM_DIR_NAME,
                               image_path.split("/")[-3], image_path.split("/")[-2])
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, image_path.split("/")[-1][:-4] + ".txt")
        with open(out_file, "w") as f:
            f.write(output_text[0])

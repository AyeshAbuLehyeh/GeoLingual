import torch
from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration


class TextModel:
    def __init__(self, model_id: str = "Qwen/Qwen3-VL-30B-A3B-Instruct"):
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        self.model.eval()
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 150) -> str:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(self.model.device)
        out_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = out_ids[0][inputs["input_ids"].shape[1]:]
        return self.processor.decode(trimmed, skip_special_tokens=True)

    @torch.no_grad()
    def generate_batch(self, prompts: list[str], max_new_tokens: int = 150) -> list[str]:
        message_batches = [[{"role": "user", "content": [{"type": "text", "text": p}]}] for p in prompts]
        texts = [
            self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in message_batches
        ]
        self.processor.tokenizer.padding_side = "left"
        enc = self.processor.tokenizer(texts, return_tensors="pt", padding=True).to(self.model.device)
        out_ids = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        results = []
        for i in range(len(prompts)):
            trimmed = out_ids[i][enc["input_ids"].shape[1]:]
            results.append(self.processor.tokenizer.decode(trimmed, skip_special_tokens=True))
        return results

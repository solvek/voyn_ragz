"""Fine-tune TrOCR on line image + transcript pairs."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    default_data_collator,
)


class TrOCRLineDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        items: list[dict[str, str]],
        processor: TrOCRProcessor,
    ) -> None:
        self.items = items
        self.processor = processor

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        path = Path(self.items[idx]["image"]).expanduser().resolve()
        text = self.items[idx]["text"]
        image = Image.open(path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=128,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {"pixel_values": pixel_values, "labels": labels}


def run_finetune(
    rows: list[dict[str, str]],
    *,
    output_dir: Path,
    base_model: str,
    num_train_epochs: float,
    per_device_train_batch_size: int,
    learning_rate: float,
) -> None:
    processor = TrOCRProcessor.from_pretrained(base_model)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model = VisionEncoderDecoderModel.from_pretrained(base_model)
    dataset = TrOCRLineDataset(rows, processor)

    args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        save_strategy="epoch",
        logging_steps=10,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        tokenizer=processor.tokenizer,
        data_collator=default_data_collator,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))

"""Clef / Clef-Flash backend (Cloudflare decision models).

Cloudflare publishes inference code only (``joint_schema_model.py`` in the
model repo). This backend reuses that file for encoding and the forward pass,
and adds the training loop:

* the released backbone (merged) is frozen and LoRA adapters are added to every
  linear layer of the language model (the vision tower is left out);
* the joint schema head is trained in full;
* the loss is cross-entropy on the gold option of every labelled question.

Note: Clef encodes all questions of a record in one sequence and decides them
jointly, unlike Kev, whose questions cannot see each other.
"""

from __future__ import annotations

import importlib.util
import json
import math
import random
import shutil
from pathlib import Path
from types import ModuleType
from typing import Any

from .base import Backend, Prediction, label_key, strip_labels

ADAPTER_DIR = "adapter"
HEAD_FILE = "joint_head.safetensors"
HEAD_CONFIG = "joint_head_config.json"
META_FILE = "clef_checkpoint.json"


def _import_release_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("joint_schema_model", path / "joint_schema_model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def lora_target_regex(model: Any) -> str:
    """Regex for peft matching every nn.Linear under the language model."""
    import torch

    leaves = sorted(
        {
            name.rsplit(".", 1)[-1]
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.Linear)
            and "language_model" in name
            and "visual" not in name
            and not name.endswith("lm_head")
        }
    )
    if not leaves:
        raise ValueError("no linear layers found under the language model")
    return r"^(?!.*visual).*language_model.*\.(" + "|".join(leaves) + r")$"


def model_input(record: dict[str, Any], with_labels: bool) -> dict[str, Any]:
    questions = record["questions"] if with_labels else strip_labels(record["questions"])
    return {"id": record.get("id", "unknown"), "state": record["state"], "questions": questions}


def gold_indices(encoded: Any, record: dict[str, Any]) -> list[int]:
    """Index of the gold option inside each encoded question's ``option_ids``."""
    indices = []
    for question in encoded.questions:
        key = label_key(record["questions"][question.question_id])
        if key not in question.option_ids:
            raise ValueError(f"{record.get('id')}: label {key!r} not in options {question.option_ids}")
        indices.append(question.option_ids.index(key))
    return indices


def batch_loss(logits: list[list[Any]], targets: list[list[int]]) -> Any:
    """Mean cross-entropy over every question in the batch."""
    import torch
    import torch.nn.functional as functional

    losses = []
    for record_logits, record_targets in zip(logits, targets):
        for question_logits, target in zip(record_logits, record_targets):
            target_tensor = torch.tensor(target, device=question_logits.device)
            losses.append(functional.cross_entropy(question_logits.float().unsqueeze(0), target_tensor.unsqueeze(0)))
    return torch.stack(losses).mean()


class ClefBackend(Backend):
    def __init__(self, model_config: dict[str, Any], device: str = "cuda"):
        super().__init__(model_config)
        self.device = device

    # ----- loading -------------------------------------------------------------
    def _release_path(self) -> Path:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(self.config["checkpoint"], revision=self.config.get("revision")))

    def _load_release(self) -> tuple[ModuleType, Any, Any, Path]:
        import torch

        path = self._release_path()
        module = _import_release_module(path)
        dtype = torch.bfloat16 if self.train_config.get("backbone_dtype", "bf16") == "bf16" else torch.float32
        model, processor = module.load_release_model(path, device=self.device, dtype=dtype)
        head_dtype = torch.float32 if self.train_config.get("head_dtype", "fp32") == "fp32" else dtype
        model.head.to(dtype=head_dtype)
        return module, model, processor, path

    def _encode(self, module: ModuleType, tokenizer: Any, record: dict[str, Any], with_labels: bool) -> Any:
        return module.encode_record(
            tokenizer,
            model_input(record, with_labels),
            max_length=int(self.train_config.get("max_length", 16384)),
            max_state_tokens=self.train_config.get("max_state_tokens"),
        )

    def _autocast(self):
        import torch

        enabled = self.device.startswith("cuda") and self.train_config.get("backbone_dtype", "bf16") == "bf16"
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=enabled)

    # ----- training ------------------------------------------------------------
    def train(
        self,
        records: list[dict[str, Any]],
        out_dir: Path,
        seed: int,
        replay: int = 0,
        replay_records: list[dict[str, Any]] | None = None,
    ) -> Path:
        import torch
        from peft import LoraConfig, get_peft_model

        cfg = self.train_config
        random.seed(seed)
        torch.manual_seed(seed)

        module, model, processor, release_path = self._load_release()
        backbone = model.language_model
        if cfg.get("gradient_checkpointing", True):
            backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            backbone.enable_input_require_grads()
        lora = LoraConfig(
            r=int(cfg.get("lora_rank", 16)),
            lora_alpha=int(cfg.get("lora_alpha", 32)),
            lora_dropout=float(cfg.get("lora_dropout", 0.0)),
            target_modules=lora_target_regex(backbone),
            bias="none",
        )
        model.language_model = get_peft_model(backbone, lora)
        for parameter in model.head.parameters():
            parameter.requires_grad_(True)

        train_records = list(records)
        if replay and replay_records:
            rng = random.Random(seed)
            train_records += rng.sample(replay_records, min(replay, len(replay_records)))

        tokenizer = processor.tokenizer
        examples = []
        for record in train_records:
            encoded = self._encode(module, tokenizer, record, with_labels=True)
            examples.append((encoded, gold_indices(encoded, record)))

        stats = self._fit(module, model, tokenizer.pad_token_id, examples, seed)

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        model.language_model.save_pretrained(out_dir / ADAPTER_DIR)
        from safetensors.torch import save_file

        save_file({k: v.detach().cpu().contiguous() for k, v in model.head.state_dict().items()}, out_dir / HEAD_FILE)
        shutil.copy(release_path / HEAD_CONFIG, out_dir / HEAD_CONFIG)
        meta = {
            "checkpoint": self.config["checkpoint"],
            "revision": self.config.get("revision"),
            "train": cfg,
            "seed": seed,
            "records": len(train_records),
            "replay": min(replay, len(replay_records or [])) if replay else 0,
            **stats,
        }
        (out_dir / META_FILE).write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        return out_dir

    def _fit(self, module: ModuleType, model: Any, pad_token_id: int, examples: list, seed: int) -> dict[str, Any]:
        import torch

        cfg = self.train_config
        batch_size = int(cfg.get("batch", 2))
        accum = int(cfg.get("accum", 1))
        epochs = int(cfg.get("epochs", 1))
        device = torch.device(self.device)

        lora_params = [p for n, p in model.language_model.named_parameters() if p.requires_grad]
        head_params = [p for p in model.head.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            [
                {"params": lora_params, "lr": float(cfg.get("lr", 2e-5))},
                {"params": head_params, "lr": float(cfg.get("head_lr", cfg.get("lr", 2e-5)))},
            ],
            weight_decay=float(cfg.get("weight_decay", 0.0)),
        )
        steps_per_epoch = math.ceil(len(examples) / (batch_size * accum))
        total_steps = max(1, steps_per_epoch * epochs)
        warmup = int(total_steps * float(cfg.get("warmup_ratio", 0.0)))

        def schedule(step: int) -> float:
            if step < warmup:
                return (step + 1) / max(1, warmup)
            return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        model.train()
        rng = random.Random(seed)
        losses: list[float] = []
        step = 0
        for _ in range(epochs):
            order = list(range(len(examples)))
            rng.shuffle(order)
            batches = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
            for batch_index, batch in enumerate(batches):
                encoded = [examples[i][0] for i in batch]
                targets = [examples[i][1] for i in batch]
                inputs = module.collate_records(encoded, pad_token_id, device)
                with self._autocast():
                    logits = model(inputs)
                loss = batch_loss(logits, targets)
                (loss / accum).backward()
                losses.append(float(loss.detach()))
                last = batch_index == len(batches) - 1
                if (batch_index + 1) % accum == 0 or last:
                    torch.nn.utils.clip_grad_norm_(lora_params + head_params, float(cfg.get("max_grad_norm", 1.0)))
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    step += 1
        tail = losses[-50:]
        return {"optimizer_steps": step, "final_loss": sum(tail) / max(1, len(tail))}

    # ----- prediction ----------------------------------------------------------
    def predict(self, checkpoint: Path | None, records: list[dict[str, Any]]) -> list[Prediction]:
        import torch

        module, model, processor, _ = self._load_release()
        if checkpoint is not None:
            from peft import PeftModel
            from safetensors.torch import load_file

            checkpoint = Path(checkpoint)
            model.language_model = PeftModel.from_pretrained(model.language_model, checkpoint / ADAPTER_DIR)
            head_state = load_file(checkpoint / HEAD_FILE)
            model.head.load_state_dict(head_state, strict=True)
        model.eval()
        return self._predict_with(module, model, processor.tokenizer, records)

    def _predict_with(self, module: ModuleType, model: Any, tokenizer: Any, records: list[dict[str, Any]]) -> list[Prediction]:
        import torch

        device = torch.device(self.device)
        batch_size = int(self.train_config.get("eval_batch", self.train_config.get("batch", 2)))
        predictions: list[Prediction] = []
        with torch.inference_mode():
            for start in range(0, len(records), batch_size):
                chunk = records[start : start + batch_size]
                encoded = [self._encode(module, tokenizer, record, with_labels=False) for record in chunk]
                inputs = module.collate_records(encoded, tokenizer.pad_token_id, device)
                with self._autocast():
                    logits = model(inputs)
                for record, enc, record_logits in zip(chunk, encoded, logits):
                    answers = {}
                    for question, question_logits in zip(enc.questions, record_logits):
                        log_probs = question_logits.float().log_softmax(-1).tolist()
                        answers[question.question_id] = dict(zip(question.option_ids, log_probs))
                    predictions.append({"id": record["id"], "answers": answers})
        return predictions

#!/usr/bin/env python3
"""Export the fine-tuned AdaptMem checkpoint to a single-graph ONNX file.

The graph embeds BERT + mean pooling + L2 normalize (sentence-transformers
module 1_Pooling mean + 2_Normalize), so runtime needs only onnxruntime +
tokenizers: same vectors as the torch path (cosine 1.0, maxdiff ~2e-7) at
~110MB RSS instead of ~590MB.

Usage (once per checkpoint):
    python scripts/export_adaptmem_onnx.py [MODEL_DIR] [OUT.onnx]

Writes OUT.onnx plus OUT.onnx.data (external weights). Runtime picks it up
automatically: _build_encoder in mnemonics/ingest.py prefers "<dir>.onnx"
next to an AdaptMem checkpoint directory.
"""
from __future__ import annotations

import sys

import torch
from transformers import AutoModel, AutoTokenizer


class _Pooled(torch.nn.Module):
    """BERT -> mean pooling (attention-masked) -> L2 normalize."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        embs = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(embs.size()).float()
        summed = (embs * mask).sum(1)
        denom = mask.sum(1).clamp(min=1e-9)
        return torch.nn.functional.normalize(summed / denom, p=2, dim=1)


def main() -> None:
    src = sys.argv[1] if len(sys.argv) > 1 else "/Users/macmini/.mnemonics/adaptmem-model/model"
    out = sys.argv[2] if len(sys.argv) > 2 else src.rstrip("/") + ".onnx"

    AutoTokenizer.from_pretrained(src)  # validates the tokenizer before export
    model = AutoModel.from_pretrained(src, attn_implementation="eager")
    model.eval()
    pooled = _Pooled(model).eval()

    ids = torch.ones(1, 8, dtype=torch.long)
    attn = torch.ones(1, 8, dtype=torch.long)
    with torch.no_grad():
        torch.onnx.export(
            pooled,
            (ids, attn),
            out,
            input_names=["input_ids", "attention_mask"],
            output_names=["embeddings"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "embeddings": {0: "batch"},
            },
            opset_version=14,
        )
    print(f"exported: {out} (+ .onnx.data)")


if __name__ == "__main__":
    main()

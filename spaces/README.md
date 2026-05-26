---
title: Comment Filtering Demo
emoji: 🛡️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "5.0.0"
app_file: app.py
pinned: false
license: apache-2.0
---

Live demo for [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) fine-tuned for
Indonesian/English chat moderation. Returns a structured JSON moderation decision
with label, severity, and confidence score.

Runs on ZeroGPU (shared A100) — inference uses `@spaces.GPU` for on-demand GPU allocation.
The Space currently serves the base Qwen3-1.7B as a placeholder; the fine-tuned weights
can be swapped in by setting the `HUB_REPO` env variable in Space settings.

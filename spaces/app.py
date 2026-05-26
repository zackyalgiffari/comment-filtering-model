"""Gradio demo for the Qwen3-1.7B comment moderation model."""

from __future__ import annotations

import json
import os
import re

import gradio as gr
import spaces
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_HUB_REPO = "Qwen/Qwen3-1.7B"
HUB_REPO = os.getenv("HUB_REPO", DEFAULT_HUB_REPO)
USING_BASE_MODEL = HUB_REPO == DEFAULT_HUB_REPO

SYSTEM_PROMPT = (
    "You are a live-stream chat moderation model for an OTT platform. "
    "Classify the user comment using only the supported taxonomy and return only valid JSON "
    "with keys flagged, labels, severity, confidence, and action."
)

LABEL_EMOJI = {
    "safe": "✅",
    "hate_or_harassment": "🚫",
    "sexual": "🔞",
    "violence_or_threat": "⚠️",
    "spam_or_scam": "📵",
    "profanity": "🤬",
    "self_harm": "🆘",
}

tokenizer: AutoTokenizer | None = None
model: AutoModelForCausalLM | None = None


def load_model() -> None:
    global tokenizer, model
    if model is not None:
        return
    tokenizer = AutoTokenizer.from_pretrained(HUB_REPO)
    model = AutoModelForCausalLM.from_pretrained(
        HUB_REPO,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()


def strip_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


@spaces.GPU(duration=60)
def moderate(comment: str, language: str) -> tuple[str, dict]:
    if not comment.strip():
        return "Please enter a comment.", {}

    load_model()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Comment language: {language}\nComment: {comment}"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            inputs,
            max_new_tokens=96,
            temperature=None,
            do_sample=False,
        )

    new_tokens = output_ids[0][inputs.shape[-1]:]
    raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    raw_text = strip_think_tags(raw_text)

    try:
        decision = json.loads(raw_text)
    except json.JSONDecodeError:
        return f"Model returned invalid JSON:\n{raw_text}", {}

    labels = decision.get("labels", [])
    label_display = " ".join(LABEL_EMOJI.get(lbl, lbl) for lbl in labels)
    flagged = decision.get("flagged", False)
    severity = decision.get("severity", "—")
    confidence = decision.get("confidence", 0.0)

    if flagged:
        summary = f"**Flagged** {label_display}  |  Severity: `{severity}`  |  Confidence: `{confidence:.0%}`"
    else:
        summary = f"**Safe** ✅  |  Confidence: `{confidence:.0%}`"

    return summary, decision


EXAMPLES = [
    ["Hei semuanya, selamat datang di stream!", "id"],
    ["Dasar bodoh! Kamu sama sekali nggak bisa main!", "id"],
    ["Buy followers now at fake-site.com!! 100% real!", "en"],
    ["Great stream, keep it up everyone!", "en"],
    ["Gue mau bunuh diri aja rasanya...", "id"],
    ["This game is trash and you're trash too", "en"],
]

with gr.Blocks(title="Comment Filtering Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🛡️ Comment Filtering Demo
        Fine-tuned **Qwen3-1.7B** for Indonesian/English live-stream chat moderation.
        Enter a comment to get a structured moderation decision.
        """
    )
    if USING_BASE_MODEL:
        gr.Markdown(
            "> ⚠️ **Placeholder mode** — currently serving the base "
            "`Qwen/Qwen3-1.7B` (not yet fine-tuned). Outputs may not strictly follow "
            "the moderation JSON schema. Set the `HUB_REPO` env var in Space settings "
            "to load the fine-tuned weights once they are published."
        )

    with gr.Row():
        with gr.Column(scale=2):
            comment_input = gr.Textbox(
                label="Comment",
                placeholder="Type a comment to moderate...",
                lines=3,
            )
            language_input = gr.Radio(
                choices=["id", "en", "mixed"],
                value="id",
                label="Language",
            )
            submit_btn = gr.Button("Moderate", variant="primary")

        with gr.Column(scale=3):
            summary_output = gr.Markdown(label="Decision summary")
            json_output = gr.JSON(label="Full decision")

    gr.Examples(
        examples=EXAMPLES,
        inputs=[comment_input, language_input],
        label="Try these examples",
    )

    submit_btn.click(
        fn=moderate,
        inputs=[comment_input, language_input],
        outputs=[summary_output, json_output],
    )
    comment_input.submit(
        fn=moderate,
        inputs=[comment_input, language_input],
        outputs=[summary_output, json_output],
    )

    gr.Markdown(
        """
        ---
        **Model:** [`{hub_repo}`](https://huggingface.co/{hub_repo}) &nbsp;|&nbsp;
        **Base:** [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) &nbsp;|&nbsp;
        **License:** Apache-2.0
        """.format(hub_repo=HUB_REPO)
    )

if __name__ == "__main__":
    demo.launch()

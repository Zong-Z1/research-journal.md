import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ── Change this to swap in a different model later ──────────────────────────
MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"

# ── Load model once at startup (CPU-only for free Hugging Face Spaces) ───────
print(f"Loading model: {MODEL_ID} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model     = AutoModelForCausalLM.from_pretrained(MODEL_ID)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=-1,              # -1 = CPU
)
print("Model ready.")

# ── A short system prefix that nudges the model toward a news-style tone ─────
NEWS_PREFIX = (
    "You are a professional news journalist. "
    "Write in a clear, factual, informative style. "
    "Continue the following news story:\n\n"
)


def generate_news(prompt, temperature, top_p, max_new_tokens):
    """Run inference and return the generated text."""
    # Guard against a temperature of exactly 0 (would break sampling)
    temperature = max(float(temperature), 0.01)

    full_prompt = NEWS_PREFIX + prompt.strip()

    result = generator(
        full_prompt,
        max_new_tokens=int(max_new_tokens),
        temperature=temperature,
        top_p=float(top_p),
        do_sample=True,
        num_return_sequences=1,
        return_full_text=False,   # only return the newly generated tokens
    )

    generated = result[0]["generated_text"].strip()
    return generated


# ── Gradio UI ────────────────────────────────────────────────────────────────
with gr.Blocks(title="News Draft Generator") as demo:

    gr.Markdown(
        """
        # 📰 News Draft Generator
        Type the start of a news sentence and let the model finish it in a
        journalistic style.

        > **Tip:** Lower temperature values (0.2 – 0.6) usually produce more
        > focused, information-style writing. Higher values add creativity but
        > can drift off-topic.
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            prompt_box = gr.Textbox(
                label="News Prompt",
                placeholder="Breaking news: scientists discovered that",
                lines=3,
            )
            submit_btn = gr.Button("Generate Draft", variant="primary")

        with gr.Column(scale=1):
            temperature_slider = gr.Slider(
                minimum=0.1, maximum=2.0, value=0.5, step=0.05,
                label="Temperature  (lower = more focused)",
            )
            top_p_slider = gr.Slider(
                minimum=0.1, maximum=1.0, value=0.9, step=0.05,
                label="Top-p  (nucleus sampling)",
            )
            max_tokens_slider = gr.Slider(
                minimum=20, maximum=200, value=100, step=10,
                label="Max New Tokens",
            )

    output_box = gr.Textbox(
        label="Generated News Draft",
        lines=6,
        interactive=False,
    )

    # Wire up the button
    submit_btn.click(
        fn=generate_news,
        inputs=[prompt_box, temperature_slider, top_p_slider, max_tokens_slider],
        outputs=output_box,
    )

    # Example prompts
    gr.Examples(
        examples=[
            ["Breaking news: scientists discovered that"],
            ["A new report released today found that students"],
            ["The conference on artificial intelligence announced"],
            ["Researchers at the university published a study showing"],
            ["This week in technology: the biggest story is"],
        ],
        inputs=prompt_box,
        label="Example Prompts  (click to load)",
    )

    gr.Markdown(
        f"*Model: `{MODEL_ID}` · Running on CPU*"
    )

if __name__ == "__main__":
    demo.launch()

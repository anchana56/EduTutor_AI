# ==============================
# 1. Install Dependencies
# ==============================
!pip install torch transformers gradio -q

# ==============================
# 2. Import Libraries
# ==============================
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==============================
# 3. Load Model & Tokenizer
# ==============================
model_name = "ibm-granite/granite-3.3-2b-instruct"  # ✅ Correct latest model

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)

# Ensure pad token exists
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ==============================
# 4. Helper Functions
# ==============================
def generate_response(prompt, max_length=512):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.replace(prompt, "").strip()

# Existing functionalities
def concept_explanation(concept):
    prompt = f"Explain the concept of {concept} in detail with examples:"
    return generate_response(prompt, max_length=800)

def quiz_generator(concept):
    prompt = f"Generate 5 quiz questions about {concept} with different question types (multiple choice, true/false, short answer). At the end, provide all the answers in a separate ANSWERS section."
    return generate_response(prompt, max_length=1000)

# New functionalities
def text_summarizer(text):
    prompt = f"Summarize the following text in simple terms:\n\n{text}"
    return generate_response(prompt, max_length=600)

def question_from_notes(notes, question):
    prompt = f"Based on the following notes:\n\n{notes}\n\nAnswer this question: {question}"
    return generate_response(prompt, max_length=600)

# ==============================
# 5. Gradio UI
# ==============================
with gr.Blocks() as app:
    gr.Markdown("# 🎓 Educational AI Assistant (IBM Granite)")

    with gr.Tabs():
        # Concept Explanation
        with gr.TabItem("Concept Explanation"):
            concept_input = gr.Textbox(label="Enter a concept", placeholder="e.g., machine learning")
            explain_btn = gr.Button("Explain")
            explanation_output = gr.Textbox(label="Explanation", lines=10)
            explain_btn.click(concept_explanation, inputs=concept_input, outputs=explanation_output)

        # Quiz Generator
        with gr.TabItem("Quiz Generator"):
            quiz_input = gr.Textbox(label="Enter a topic", placeholder="e.g., physics")
            quiz_btn = gr.Button("Generate Quiz")
            quiz_output = gr.Textbox(label="Quiz Questions", lines=15)
            quiz_btn.click(quiz_generator, inputs=quiz_input, outputs=quiz_output)

        # Summarizer
        with gr.TabItem("Summarizer"):
            text_input = gr.Textbox(label="Paste text to summarize", lines=8)
            summarize_btn = gr.Button("Summarize")
            summary_output = gr.Textbox(label="Summary", lines=8)
            summarize_btn.click(text_summarizer, inputs=text_input, outputs=summary_output)

        # Ask Questions from Notes
        with gr.TabItem("Ask from Notes"):
            notes_input = gr.Textbox(label="Paste your notes", lines=10, placeholder="e.g., biology chapter notes")
            question_input = gr.Textbox(label="Enter your question", placeholder="e.g., What is photosynthesis?")
            ask_btn = gr.Button("Get Answer")
            answer_output = gr.Textbox(label="Answer", lines=6)
            ask_btn.click(question_from_notes, inputs=[notes_input, question_input], outputs=answer_output)

# ==============================
# 6. Launch App
# ==============================
app.launch(share=True)
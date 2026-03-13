# serve/app.py
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextIteratorStreamer
import torch, threading, os
from transformers import AutoModelForCausalLM, AutoTokenizer

app       = FastAPI()
templates = Jinja2Templates(directory="serve/templates")

MODEL_DIR = "model/gpt2_grpo_merged"
device    = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model from {MODEL_DIR} on {device}...")
tok       = GPT2Tokenizer.from_pretrained("gpt2")
# tok   = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
tok.pad_token = tok.eos_token
model     = GPT2LMHeadModel.from_pretrained(MODEL_DIR).to(device).eval()
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
print(f"Model ready on {device}. Params: {sum(p.numel() for p in model.parameters()):,}")


# ── Request schema — Pydantic handles JSON parsing correctly ──────────────────
class GenerateRequest(BaseModel):
    prompt:      str   = "Hello"
    max_tokens:  int   = 100
    temperature: float = 0.8


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/generate")
async def generate(req: GenerateRequest):   # ← Pydantic model, not raw Request
    formatted = f"\n<|user|>\n{req.prompt.strip()}\n<|assistant|>\n"
    input_ids = tok.encode(formatted, return_tensors="pt").to(device)

    streamer   = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(
        input_ids      = input_ids,
        max_new_tokens = req.max_tokens,
        temperature    = req.temperature,
        top_k          = 50,
        do_sample      = True,
        pad_token_id   = tok.eos_token_id,
        eos_token_id   = tok.eos_token_id,
        streamer       = streamer,
    )

    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    def token_stream():
        for token in streamer:
            yield token

    return StreamingResponse(token_stream(), media_type="text/plain")

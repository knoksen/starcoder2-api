from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the model (this may take a while the first time!)
checkpoint = "bigcode/starcoder2-3b"  # Change to 7b or 15b if desired and hardware allows
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(
    checkpoint,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

# Define the FastAPI app
app = FastAPI()

# Request/response structure
class CodeRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 128

@app.post("/generate")
def generate_code(req: CodeRequest):
    inputs = tokenizer(req.prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=req.max_new_tokens, do_sample=False)
    code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"code": code}

import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


os.environ['HF_TOKEN']="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"
os.environ['HUGGINGFACEHUB_API_TOKEN']="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"


model_id = "meta-llama/Llama-3.1-8B-Instruct" 

tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)

sentence= "How many singers do we have?"
prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\nParaphrase: {sentence}.<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

# 🧼 Clean decode (skip prompt tokens)
generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)

# ✅ Print only the paraphrased sentence
print("paraphrased sentence: ",generated_text.strip())

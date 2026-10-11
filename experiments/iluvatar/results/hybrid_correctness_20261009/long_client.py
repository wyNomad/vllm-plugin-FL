#!/usr/bin/env python3
"""Send a deterministic roughly-16k-token prompt through the chat endpoint."""

import json
import sys
import urllib.request

from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("/workspace/MiniCPM5-2B")
filler = "Background record: this sentence contains no task instruction. "
question = "\nEnd of context. What is 2 + 2? State the answer as a number."


def build_prompt(repeats):
    prompt = filler * repeats + question
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True,
    )
    # This model's tokenizer can return a mapping for tokenize=True; len()
    # would then count fields rather than tokens and massively overshoot.
    return prompt, len(tokenizer.encode(rendered, add_special_tokens=False))


lower, upper = 0, 10000
while lower < upper:
    middle = (lower + upper + 1) // 2
    _, count = build_prompt(middle)
    if count <= 16384:
        lower = middle
    else:
        upper = middle - 1
prompt, input_tokens = build_prompt(lower)
print(f"input_tokens={input_tokens}; filler_repeats={lower}", file=sys.stderr)
if not 16000 <= input_tokens <= 16384:
    raise RuntimeError(f"expected roughly 16k input tokens, got {input_tokens}")

payload = {
    "model": "minicpm", "messages": [{"role": "user", "content": prompt}],
    "temperature": 0, "max_tokens": 128, "seed": 42,
}
request = urllib.request.Request(
    "http://127.0.0.1:9031/v1/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
)
with urllib.request.urlopen(request, timeout=250) as response:
    print(response.read().decode())

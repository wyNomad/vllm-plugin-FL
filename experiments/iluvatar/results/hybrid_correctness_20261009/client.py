#!/usr/bin/env python3
"""Send a single deterministic request from inside the disposable container."""

import json
import urllib.request


prompt = (
    "What is the distance, in units, between the points $(2, -6)$ and "
    "$(-4, 3)$? Express your answer in simplest radical form.\n"
    "Please reason step by step, and put your final answer within \\boxed{}."
)
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

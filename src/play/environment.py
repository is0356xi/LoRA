import os
from transformers import AutoTokenizer

token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
print(token)

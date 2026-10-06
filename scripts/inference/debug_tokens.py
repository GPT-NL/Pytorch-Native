import json

import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint_dir = (
    # "/INSERT-GPT-NL-TRAINING-DIR/converted_checkpoints/gpt-nl-26B-nodes-22-mbs-12-gas-3-13294555/step44550/"
    # "/INSERT-GPT-NL-TRAINING-DIR/converted_checkpoints/gpt-nl-26B-nodes-22-mbs-12-gas-3-13594524/step48400/"
    # "/INSERT-GPT-NL-TRAINING-DIR/converted_checkpoints/gpt-nl-26B-nodes-22-mbs-12-gas-3-13594524/step51260/"
    # "/INSERT-GPT-NL-TRAINING-DIR/converted_checkpoints/gpt-nl-26B-nodes-22-mbs-12-gas-3-14003231/step55330"
    "/INSERT-GPT-NL-TRAINING-DIR/converted_checkpoints/gpt-nl-26B-nodes-22-mbs-12-gas-3-14376145/step77440"
)


messages = [
    "The meaning of life is",
    # "What is a potato?",
    # "Is an airplane faster than a car?",
    # "What is the boiling point of water?",
    # "De zin van het leven is",
    # "Wat is een aardappel?",
    # "Is een vliegtuig sneller dan een auto?",
    # "Wat is het kookpunt van water?",
]

olmo = AutoModelForCausalLM.from_pretrained(checkpoint_dir, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

input_text = messages[0]
inputs = tokenizer([input_text], return_tensors="pt", return_token_type_ids=False)
inputs = {k: v.to("cuda") for k, v in inputs.items()}
response = olmo.generate(**inputs, max_new_tokens=4096, temperature=0.7, do_sample=True, top_k=50, top_p=0.95)
# find EOS token
res_np = response.cpu().numpy()
array = res_np[0]
indices = np.where(array == tokenizer.eos_token_id)
indices[0].shape

res_string = tokenizer.batch_decode(response, skip_special_tokens=True)[0]
res_string_raw = tokenizer.batch_decode(response, skip_special_tokens=False)[0]
res_string_raw

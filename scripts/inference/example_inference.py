# # model needs to be converted with `scripts/inference/convert_checkpoint.job`
checkpoint_dir = "/INSERT-GPT-NL-TRAINING-DIR/converted_checkpoints"
checkpoint_dir = (
    # "/INSERT-GPT-NL-TRAINING-DIR/converted_checkpoints/gpt-nl-26B-nodes-22-mbs-12-gas-3-14003231/step54120"
    # "/INSERT-GPT-NL-TRAINING-DIR/converted_checkpoints/gpt-nl-26B-nodes-22-mbs-12-gas-3-13594524/step51260/"
    "/INSERT-GPT-NL-TRAINING-DIR/converted_checkpoints/gpt-nl-26B-nodes-22-mbs-12-gas-3-14003231/step55330/"
)

messages = [
    "The meaning of life is",
    "What is a potato?",
    "Is an airplane faster than a car?",
    "What is the boiling point of water?",
    "De zin van het leven is",
    "Wat is een aardappel?",
    "Is een vliegtuig sneller dan een auto?",
    "Wat is het kookpunt van water?",
]

# with the pipeline
pipe = pipeline(task="text-generation", model=checkpoint_dir, device_map="auto")
outputs = pipe(
    messages,
    max_new_tokens=50,
    do_sample=True,  # False sets to greedy decoding (always most probable token)
    top_k=50,
    top_p=0.95,
    temperature=0.9,
)
assert isinstance(outputs, list)
for q, a in zip(messages, outputs):
    a_text = a[0]["generated_text"]
    print(f"{q}\t-->\t{a_text[len(q):]}")

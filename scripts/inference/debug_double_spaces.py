# run this interactively: python -i scripts/inference/debug_double_spaces.py

# trying to debug why the model generates multiple spaces

# summary:
# We can see the root cause by setting `skip_special_tokens=False` during decode.
# All the cases happen with characters around token '\n' (id: 27).
# Preceding tokens are:
# - 127675: "_"
# - 9 "▁▁"
# - 10 "▁▁▁"
# Sometimes 127675 also follows '\n'

import json

from transformers import AutoModelForCausalLM, AutoTokenizer

# # model needs to be converted with `scripts/inference/convert_checkpoint.job`
checkpoint_dir = "/INSERT-GPT-NL-TRAINING-DIR/pn_checkpoints_keep/huggingface-models/step17380"

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

olmo = AutoModelForCausalLM.from_pretrained(checkpoint_dir, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

input_text = messages[-2]
inputs = tokenizer([input_text], return_tensors="pt", return_token_type_ids=False)
inputs = {k: v.to("cuda") for k, v in inputs.items()}

# find a case with double spaces
res_string_raw = response = None
for i in range(10):
    response = olmo.generate(**inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
    res_string = tokenizer.batch_decode(response, skip_special_tokens=True)[0]
    res_string_raw = tokenizer.batch_decode(response, skip_special_tokens=False)[0]
    # print(res_string_raw)
    # diff = difflib.ndiff(res_string, res_string_raw)
    if "  " in res_string:
        break

assert res_string_raw, "no string with double spaces"
# if printing, \n gets rendered
# print(res_string_raw)
print(repr(res_string_raw))
# '<s> Is een vliegtuig sneller dan een auto? Antwoord: \n De ontwikkeling van een vliegtuig is niet anders dan die van een auto: deze wordt minder snel \n duurder.  \n  \n Bron: SEO Economisch Onderzoek  \n 3.3.2 Bevoorschotting van vliegtickets \n Door wie en waarom?   \n Luchtvaartmaatschappijen zijn zelf verantwoordelijk om hun vliegtickets in te zetten op basis van \n hun eigen begroting (art. 2.2.1, lid 1, sub c Omgevingswet). Voor vliegtuigen die'
assert response is not None
token_ids = response[0]
decoded_tokens = [tokenizer.decode([token], clean_up_tokenization_spaces=False) for token in token_ids]
decoded_tokens_cleanup = [tokenizer.decode([token]) for token in token_ids]
assert len(decoded_tokens) == len(decoded_tokens_cleanup), "tokenization_spaces makes a difference"
together = [f"{t.item()}: {repr(s)}" for t, s in zip(token_ids, decoded_tokens)]
print(json.dumps(together, indent=2))
# token ids can be seen in /INSERT-GPT-NL-TRAINING-DIR/tokenizer/gpt-nl-128k-1.0/hf/tokenizer.json
# to format outputs, can do replace: `\), \(` into `),\n    (`
# [
#     (1, '<s>'),
#     (2111, 'Is'),
#     (492, 'een'),
#     (37348, 'vliegtuig'),
#     (24644, 'sneller'),
#     (868, 'dan'),
#     (492, 'een'),
#     (3971, 'auto'),
#     (127758, '?'),
#     (8806, 'Antwoord'),
#     (127714, ':'),
#     (127675, ''),               # 127675 is "_"
#     (27, '\n'),                 # \n
#     (564, 'De'),
#     (3536, 'ontwikkeling'),
#     (425, 'van'),
#     (492, 'een'),
#     (37348, 'vliegtuig'),
#     (476, 'is'),
#     (602, 'niet'),
#     (3066, 'anders'),
#     (868, 'dan'),
#     (546, 'die'),
#     (425, 'van'),
#     (492, 'een'),
#     (3971, 'auto'),
#     (127714, ':'),
#     (751, 'deze'),
#     (740, 'wordt'),
#     (2368, 'minder'),
#     (5016, 'snel'),
#     (127675, ''),               # 127675 is "_"
#     (27, '\n'),                 # \n
#     (43673, 'duurder'),
#     (127693, '.'),
#     (9, ' '),                   # 9 is "▁▁"
#     (27, '\n'),                 # \n
#     (9, ' '),                   # 9 is "▁▁"
#     (27, '\n'),                 # \n
#     (17998, 'Bron'),
#     (127714, ':'),
#     (69341, 'SEO'),
#     (49626, 'Economisch'),
#     (12292, 'Onderzoek'),
#     (9, ' '),                   # 9 is "▁▁"
#     (27, '\n'),                 # \n
#     (127675, ''),               # 127675 is "_"
#     (127722, '3'),
#     (127693, '.'),
#     (127722, '3'),
#     (127693, '.'),
#     (127707, '2'),
#     (1176, 'Be'),
#     (895, 'voor'),
#     (81654, 'schotting'),
#     (425, 'van'),
#     (82278, 'vliegt'),
#     (84627, 'ickets'),
#     (127675, ''),               # 127675 is "_"
#     (27, '\n'),                 # \n
#     (3264, 'Door'),
#     (1834, 'wie'),
#     (459, 'en'),
#     (5113, 'waarom'),
#     (127758, '?'),
#     (10, '  '),                 # 10 is "▁▁▁"
#     (27, '\n'),                 # \n
#     (50448, 'Luchtvaart'),
#     (34374, 'maatschappijen'),
#     (610, 'zijn'),
#     (1933, 'zelf'),
#     (3497, 'verantwoord'),
#     (552, 'elijk'),
#     (611, 'om'),
#     (1770, 'hun'),
#     (82278, 'vliegt'),
#     (84627, 'ickets'),
#     (419, 'in'),
#     (495, 'te'),
#     (5517, 'zetten'),
#     (486, 'op'),
#     (1975, 'basis'),
#     (425, 'van'),
#     (127675, ''),               # 127675 is "_"
#     (27, '\n'),                 # \n
#     (1770, 'hun'),
#     (1852, 'eigen'),
#     (3922, 'begroting'),
#     (455, '('),
#     (584, 'art'),
#     (127693, '.'),
#     (127675, ''),
#     (127707, '2'),
#     (127693, '.'),
#     (127707, '2'),
#     (127693, '.'),
#     (127703, '1'),
#     (127698, ','),
#     (1248, 'lid'),
#     (127675, ''),
#     (127703, '1'),
#     (127698, ','),
#     (975, 'sub'),
#     (430, 'c'),
#     (16378, 'Omgevingswet'),
#     (604, ').'),
#     (1152, 'Voor'),
#     (44487, 'vliegtuigen'),
#     (546, 'die')
# ]


# another example
# '<s> Is een vliegtuig sneller dan een auto? (T/m 30 jaar, of T 50 jaar) \n  \n 4. Voor de bepaling van de levensduur per type vliegtuig is er geen \n betrouwbare methode. Kan er een overzicht zijn van de \n levensduur per type vliegtuig? (T/m 27 jaar, T/m 34 jaar of T 50 jaar) \n  \n  \n  \n  \n  \n 5. Is er ruimte voor verschillende type vliegtuigen om'
# [
#     (1, '<s>'),
#     (2111, 'Is'),
#     (492, 'een'),
#     (37348, 'vliegtuig'),
#     (24644, 'sneller'),
#     (868, 'dan'),
#     (492, 'een'),
#     (3971, 'auto'),
#     (127758, '?'),
#     (455, '('),
#     (127710, 'T'),
#     (127716, '/'),
#     (127690, 'm'),
#     (127675, ''),
#     (127722, '3'),
#     (127704, '0'),
#     (1254, 'jaar'),
#     (127698, ','),
#     (439, 'of'),
#     (477, 'T'),
#     (127675, ''),
#     (127730, '5'),
#     (127704, '0'),
#     (1254, 'jaar'),
#     (127705, ')'),
#     (127675, ''),               # 127675 is "_"
#     (27, '\n'),                 # \n
#     (9, ' '),                   # 9 is "▁▁"
#     (27, '\n'),                 # \n
#     (127675, ''),
#     (127731, '4'),
#     (127693, '.'),
#     (1152, 'Voor'),
#     (408, 'de'),
#     (6072, 'bepaling'),
#     (425, 'van'),
#     (408, 'de'),
#     (37151, 'levensduur'),
#     (723, 'per'),
#     (1805, 'type'),
#     (37348, 'vliegtuig'),
#     (476, 'is'),
#     (615, 'er'),
#     (917, 'geen'),
#     (127675, ''),               # 127675 is "_"
#     (27, '\n'),                 # \n
#     (37921, 'betrouwbare'),
#     (15580, 'methode'),
#     (127693, '.'),
#     (8568, 'Kan'),
#     (615, 'er'),
#     (492, 'een'),
#     (6590, 'overzicht'),
#     (610, 'zijn'),
#     (425, 'van'),
#     (408, 'de'),
#     (127675, ''),               # 127675 is "_"
#     (27, '\n'),                 # \n
#     (37151, 'levensduur'),
#     (723, 'per'),
#     (1805, 'type'),
#     (37348, 'vliegtuig'),
#     (127758, '?'),
#     (455, '('),
#     (127710, 'T'),
#     (127716, '/'),
#     (127690, 'm'),
#     (127675, ''),
#     (127707, '2'),
#     (127741, '7'),
#     (1254, 'jaar'),
#     (127698, ','),
#     (477, 'T'),
#     (127716, '/'),
#     (127690, 'm'),
#     (127675, ''),
#     (127722, '3'),
#     (127731, '4'),
#     (1254, 'jaar'),
#     (439, 'of'),
#     (477, 'T'),
#     (127675, ''),
#     (127730, '5'),
#     (127704, '0'),
#     (1254, 'jaar'),
#     (127705, ')'),
#     (127675, ''),               # 127675 is "_"
#     (27, '\n'),                 # \n
#     (9, ' '),                   # 9 is "▁▁"
#     (27, '\n'),                 # \n
#     (9, ' '),                   # 9 is "▁▁"
#     (27, '\n'),                 # \n
#     (9, ' '),                   # 9 is "▁▁"
#     (27, '\n'),                 # \n
#     (9, ' '),                   # 9 is "▁▁"
#     (27, '\n'),                 # \n
#     (9, ' '),                   # 9 is "▁▁"
#     (27, '\n'),                 # \n
#     (127675, ''),               # 127675 is "_"
#     (127730, '5'),
#     (127693, '.'),
#     (2111, 'Is'),
#     (615, 'er'),
#     (3834, 'ruimte'),
#     (500, 'voor'),
#     (3044, 'verschillende'),
#     (1805, 'type'),
#     (44487, 'vliegtuigen'),
#     (611, 'om')
# ]

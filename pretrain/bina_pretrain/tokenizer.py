import tiktoken

_ENC = tiktoken.get_encoding("gpt2")

VOCAB_SIZE = _ENC.n_vocab
EOT = _ENC.eot_token  # 50256, used as a document separator in the pretraining token stream

# Chat-role markers for the SFT stage (module_32's "### Question:"/"### Answer:" pattern,
# extended from one prompt/response pair to arbitrary system/user/bina turns). Ordinary
# multi-token text, not special tokens added to the vocab -- keeps the vocab/tokenizer
# identical between the pretrain and SFT stages.
SYSTEM_MARKER = "\n<|system|>\n"
USER_MARKER = "\n<|user|>\n"
BINA_MARKER = "\n<|bina|>\n"


def encode(text):
    return _ENC.encode_ordinary(text)


def decode(token_ids):
    return _ENC.decode(token_ids)

from transformers import PreTrainedTokenizerFast


class EshmunTokenizer(PreTrainedTokenizerFast):
    def __init__(self, tokenizer_object=None, tokenizer_file=None, **kwargs):
        kwargs.setdefault("bos_token", "<bos>")
        kwargs.setdefault("eos_token", "<eos>")
        kwargs.setdefault("unk_token", "<unk>")
        kwargs.setdefault("pad_token", "<eos>")

        super().__init__(
            tokenizer_object=tokenizer_object, tokenizer_file=tokenizer_file, **kwargs
        )

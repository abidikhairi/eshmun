from tokenizers import Tokenizer, decoders, pre_tokenizers
from tokenizers.models import BPE
from transformers import PreTrainedTokenizerFast


class EshmunTokenizer(PreTrainedTokenizerFast):
    """Byte-level BPE tokenizer for the Eshmun models.

    ``model = BPE`` is load-bearing. transformers 5.x hands a class that
    declares its own ``__init__`` the contents of ``tokenizer.json`` as
    ``vocab``/``merges`` kwargs and expects the class to rebuild the backend,
    and it only lifts the merges out of the file for classes that declare
    their backend model. Without the declaration, ``from_pretrained`` silently
    returns a tokenizer with no merges, no pre-tokenizer and no decoder, which
    encodes every string as raw bytes instead of the tokenization the models
    were trained on. Rebuilding here mirrors the stock GPT-2 tokenizer, whose
    on-disk format this one shares.
    """

    model = BPE

    def __init__(
        self,
        tokenizer_object=None,
        tokenizer_file=None,
        vocab=None,
        merges=None,
        add_prefix_space=False,
        **kwargs,
    ):
        kwargs.setdefault("bos_token", "<bos>")
        kwargs.setdefault("eos_token", "<eos>")
        kwargs.setdefault("unk_token", "<unk>")
        kwargs.setdefault("pad_token", "<eos>")

        # The 5.x load path passes no backend object and no file, just the
        # pieces. `tokenizer_file` and `tokenizer_object` still cover direct
        # construction and 4.x-style loads, so only rebuild when neither came in.
        if tokenizer_object is None and tokenizer_file is None and vocab is not None:
            tokenizer_object = Tokenizer(
                BPE(
                    vocab=vocab,
                    merges=merges or [],
                    dropout=None,
                    continuing_subword_prefix="",
                    end_of_word_suffix="",
                    fuse_unk=False,
                )
            )
            tokenizer_object.pre_tokenizer = pre_tokenizers.ByteLevel(
                add_prefix_space=add_prefix_space
            )
            tokenizer_object.decoder = decoders.ByteLevel()

        super().__init__(
            tokenizer_object=tokenizer_object, tokenizer_file=tokenizer_file, **kwargs
        )

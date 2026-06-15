TRAIN := python scripts/eshmun-zero/train.py

# Shared hyperparameters — override from the command line if needed
# e.g.  make mha EPOCHS=20 BATCH_SIZE=16
EPOCHS             ?= 5
BATCH_SIZE         ?= 12
HIDDEN_SIZE        ?= 256
INTERMEDIATE_SIZE  ?= 512
NUM_LAYERS         ?= 12
NUM_HEADS          ?= 8
NUM_KV_HEADS       ?= 4
MAX_SEQ_LEN        ?= 512
DROPOUT            ?= 0.1
TOP_K              ?= 5
WINDOW_SIZE        ?= 10
LEARNING_RATE      ?= 2e-4
WEIGHT_DECAY       ?= 0.01
GRAD_ACCUM         ?= 12
GRAD_CLIP          ?= 1.0
WARMUP_STEPS       ?= 100
LOGGING_STEPS      ?= 50
DATA_PATH          ?= data/sequences.txt
OUTPUT_BASE        ?= data/eshmun-zero/models

COMMON_ARGS = \
	--data_path          $(DATA_PATH) \
	--num_epochs         $(EPOCHS) \
	--batch_size         $(BATCH_SIZE) \
	--hidden_size        $(HIDDEN_SIZE) \
	--intermediate_size  $(INTERMEDIATE_SIZE) \
	--num_layers         $(NUM_LAYERS) \
	--num_heads          $(NUM_HEADS) \
	--num_kv_heads       $(NUM_KV_HEADS) \
	--max_seq_len        $(MAX_SEQ_LEN) \
	--dropout            $(DROPOUT) \
	--top_k              $(TOP_K) \
	--window_size        $(WINDOW_SIZE) \
	--learning_rate      $(LEARNING_RATE) \
	--weight_decay       $(WEIGHT_DECAY) \
	--gradient_accumulation_steps $(GRAD_ACCUM) \
	--gradient_clip      $(GRAD_CLIP) \
	--warmup_steps       $(WARMUP_STEPS) \
	--logging_steps      $(LOGGING_STEPS) \
	--use_rope

.PHONY: all mha sparse sliding_window gqa compressed gated token_filter

all: mha sparse sliding_window gqa compressed gated

mha:
	$(TRAIN) $(COMMON_ARGS) --attn_impl mha --output_dir $(OUTPUT_BASE)/mha

token_filter: # DONE
	$(TRAIN) $(COMMON_ARGS) --attn_impl token_filter --output_dir $(OUTPUT_BASE)/token_filter

sparse:
	$(TRAIN) $(COMMON_ARGS) --attn_impl sparse --output_dir $(OUTPUT_BASE)/sparse

sliding_window:
	$(TRAIN) $(COMMON_ARGS) --attn_impl sliding_window --output_dir $(OUTPUT_BASE)/sliding_window

gqa:
	$(TRAIN) $(COMMON_ARGS) --attn_impl gqa --output_dir $(OUTPUT_BASE)/gqa

compressed:
	$(TRAIN) $(COMMON_ARGS) --attn_impl compressed --output_dir $(OUTPUT_BASE)/compressed

gated:
	$(TRAIN) $(COMMON_ARGS) --attn_impl gated --output_dir $(OUTPUT_BASE)/gated

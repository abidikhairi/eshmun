# Entry points for the Eshmun fine-tuning workflow.
#
#   make sync          copy this repo to the GPU server
#   make finetune      run the SFT script there
#   make pull <ckpt>   copy a checkpoint back here, e.g.
#                        make pull checkpoint-1000
#                      It lands in runs/sft/checkpoint-1000, which is
#                      gitignored and excluded from `make sync`.
#
# The settings below are defaults. Override any of them in Makefile.local
# (gitignored), or per invocation, e.g.
#
#   make finetune RECIPE=recipes/sft/other.yaml

-include Makefile.local

SSH_HOST   ?= 41.226.183.241
SSH_PORT   ?= 2224
SSH_USER   ?= kabidi
REMOTE_DIR ?= /home/kabidi/workspace/projects/eshmun
REMOTE_PY  ?= /opt/miniconda3/bin/python
RECIPE     ?= recipes/sft/sprot-protein-design-small.yaml
# Where the recipe above writes, on the server, and where `pull` copies to here.
REMOTE_RUN_DIR ?= runs/sft/sprot-protein-design-small
LOCAL_RUN_DIR  ?= runs/sft

SSH := ssh -p $(SSH_PORT) -o BatchMode=yes $(SSH_USER)@$(SSH_HOST)

.PHONY: sync finetune pull

sync:
	rsync -avz \
		--exclude='.git/' \
		--exclude='__pycache__/' --exclude='*.py[cod]' --exclude='.pytest_cache/' \
		--exclude='.venv/' --exclude='runs/' --exclude='wandb/' --exclude='gpu_server.txt' \
		-e "ssh -p $(SSH_PORT) -o BatchMode=yes" ./ \
		$(SSH_USER)@$(SSH_HOST):$(REMOTE_DIR)/

finetune:
	$(SSH) 'cd $(REMOTE_DIR) && $(REMOTE_PY) scripts/training/supervised_finetune.py --recipe $(RECIPE)'

# `make pull checkpoint-1000`. Make reads every word of the goal list as a
# target, so the checkpoint name is lifted out of MAKECMDGOALS here, and the
# no-op pattern rule below absorbs it before make can look for a rule of its own.
ifneq ($(filter pull,$(MAKECMDGOALS)),)
CHECKPOINT := $(filter-out pull,$(MAKECMDGOALS))
endif

pull:
	@test -n "$(CHECKPOINT)" || \
		{ echo "usage: make pull <checkpoint-dir>   e.g. make pull checkpoint-1000"; exit 1; }
	mkdir -p $(LOCAL_RUN_DIR)
	# optimizer.pt is the AdamW state -- about 3.3 GB of the 4.9 GB checkpoint,
	# and only needed to resume training from it.
	rsync -avz --partial \
		--exclude='optimizer.pt' \
		-e "ssh -p $(SSH_PORT) -o BatchMode=yes" \
		$(SSH_USER)@$(SSH_HOST):$(REMOTE_DIR)/$(REMOTE_RUN_DIR)/$(CHECKPOINT)/ \
		$(LOCAL_RUN_DIR)/$(CHECKPOINT)/

# Defined only while `pull` is a goal, so an undefined target anywhere else is
# still an error rather than a silent no-op.
ifneq ($(filter pull,$(MAKECMDGOALS)),)
%:
	@:
endif

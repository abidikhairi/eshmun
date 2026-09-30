# Entry points for the Eshmun fine-tuning workflow.
#
#   make sync      copy this repo to the GPU server
#   make finetune  run the SFT script there
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

SSH := ssh -p $(SSH_PORT) -o BatchMode=yes $(SSH_USER)@$(SSH_HOST)

.PHONY: sync finetune

sync:
	rsync -avz \
		--exclude='__pycache__/' --exclude='*.py[cod]' --exclude='.pytest_cache/' \
		--exclude='.venv/' --exclude='runs/' --exclude='wandb/' --exclude='gpu_server.txt' \
		-e "ssh -p $(SSH_PORT) -o BatchMode=yes" ./ \
		$(SSH_USER)@$(SSH_HOST):$(REMOTE_DIR)/

finetune:
	$(SSH) 'cd $(REMOTE_DIR) && $(REMOTE_PY) scripts/training/supervised_finetune.py --recipe $(RECIPE)'

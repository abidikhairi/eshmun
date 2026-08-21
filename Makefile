.PHONY: install typecheck clean

install:
	pip install -e .

typecheck:
	pyrefly check

clean:
	find . -not -path './.git*' -name '__pycache__' -type d -exec rm -rf {} +
	find . -not -path './.git*' -name '*.py[oc]' -delete
	rm -rf src/eshmun.egg-info .pytest_cache

-include Makefile.local

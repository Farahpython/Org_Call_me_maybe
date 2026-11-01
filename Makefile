
.PHONY: install run debug clean activate lint

VENV ?= $(if $(wildcard /goinfre),/goinfre/fadarwis/venvs/call_me_maybe,.venv)

install:
	UV_PROJECT_ENVIRONMENT=$(VENV) uv sync
	UV_PROJECT_ENVIRONMENT=$(VENV) uv pip install flake8 mypy
run:
	UV_PROJECT_ENVIRONMENT=$(VENV) uv run python -m src
debug:
	UV_PROJECT_ENVIRONMENT=$(VENV) uv run python -m pdb -m src

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf $(VENV)
activate:
	@echo "source $(VENV)/bin/activate"
lint:
	$(VENV)/bin/flake8 . --exclude=$(VENV),llm_sdk
	$(VENV)/bin/mypy . --exclude='($(VENV)|llm_sdk)' --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs	

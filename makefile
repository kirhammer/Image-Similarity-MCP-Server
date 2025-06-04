VENV_DIR := .venv
PYTHON := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip

.PHONY: install run clean

install:
	python3 -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

run:
	$(VENV_DIR)/bin/uvicorn server:app --reload

clean:
	rm -rf $(VENV_DIR)
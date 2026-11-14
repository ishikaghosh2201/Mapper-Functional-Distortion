PYTHON ?= python

.PHONY: tests
tests:
	$(PYTHON) -m pytest -q tests

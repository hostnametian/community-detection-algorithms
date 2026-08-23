PYTHON ?= python3

.PHONY: check validate prepare main reproduce

check:
	$(PYTHON) scripts/00_check_setup.py --allow-missing-data

validate:
	$(PYTHON) scripts/00_validate_algorithms.py
	$(PYTHON) scripts/00b_validate_acceleration.py

prepare:
	$(PYTHON) scripts/01_prepare_data.py

main:
	$(PYTHON) scripts/02_run_main_analysis.py

reproduce:
	$(PYTHON) scripts/08_run_all.py

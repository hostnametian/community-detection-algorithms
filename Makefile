PYTHON ?= python3

.PHONY: check validate prepare main reproduce revision-associations revision-validate

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

revision-associations:
	$(PYTHON) scripts/revision/metric_associations.py
	$(PYTHON) scripts/revision/plot_metric_correlations.py
	$(PYTHON) scripts/revision/plot_within_method_scatter.py

revision-validate:
	$(PYTHON) scripts/revision/03_validate_metrics.py
	$(PYTHON) scripts/revision/03b_planted_validation.py

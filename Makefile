PYTHON := ./venv/bin/python

clean-outputs:
	rm -rf outputs/*

gen-images:
	$(PYTHON) scripts/generate_figure_heatmap_h2l_auroc.py
	$(PYTHON) scripts/generate_figure_heatmap_h2l_tpr.py
PYTHON ?= python
EPOCHS ?= 12
PUBLIC ?=

.PHONY: all data base gru test

all: gru

data:
	$(PYTHON) -m pipeline.data_features

base: data
	$(PYTHON) -m pipeline.base_recommender $(PUBLIC)

gru: base
	$(PYTHON) -m pipeline.gru_extension --epochs $(EPOCHS) $(PUBLIC)

test:
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 $(PYTHON) -m pytest tests/test_recommender.py -q

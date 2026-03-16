PYTHON = python3

install:
	pip install -r requirements.txt

process_data:
	$(PYTHON) -m scripts.prepare_dataset

train_detector:
	$(PYTHON) -m scripts.train_detector

test_detector:
	$(PYTHON) -m scripts.test_detector

demo_detector:
	$(PYTHON) -m scripts.demo_detector

track_video:
	$(PYTHON) -m scripts.track_video

clean:
	rm -rf runs
	rm -rf outputs
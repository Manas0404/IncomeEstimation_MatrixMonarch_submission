# Income Estimation Hackathon Solution

## Structure
- `main.py`: Entrypoint to run the whole pipeline.
- `run_inference.py`: Run predictions on test/hidden set.
- `features/feat_engineering.py`: All feature creation logic.
- `models/train_model.py`: Model training code.
- `models/predict_model.py`: Inference code.
- `output/`: Output CSVs.
- `data/`: All provided data files.

## Usage

1. **Train:**
python main.py train

2. **Predict:**
python run_inference.py


Outputs will be saved in `/output/`.

## Dependencies
See `requirements.txt`.

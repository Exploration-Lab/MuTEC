## Two Step architecture for Cause Span Extraction

## MuTEC (CSE): Cause Span Extraction 
Install required libraries:
```bash
pip install -r requirements.txt
```
- `cause_span_model.py`: Cause Span Predictor model.
- `emotion_model.py`: Emotion Predictor model.
- `dataset.py`: Custom dataset.
- `config.py`: Contains the hyperparameter configurations used. Also contains the dataset locations.
- `model.py`: Contains the Two Step model implementation.
- `run_two.py`: Run script.
- `eval_two.py`: Eval Script.
- `question_answering_utils.py`: Span prediction utility functions.
- `utils_squad.py`: Evaluation metrics functions for span prediction.
- `utils.py`: Customer utility functions.
- `data/`: Directory containing data for MuTEC (CSE).

### Example Code Run
#### For Training
```bash
python run_two.py
```
#### For Evaluation
```bash 
python eval_two.py
```


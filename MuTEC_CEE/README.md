## MuTEC: Causal Emotion Entailment
Install required libraries:
```bash
pip install -r requirements.txt
```
- `config.py`: Contains the hyperparameter configurations used. Also contains the dataset locations.
- `dataset.py`: Custom Dataset script.
- `model.py`: Contains the Task2 models implemented.
- `run_cee.py`: Run script.
- `eval_cee.py`: Eval Script.
- `utils.py`: Utility functions.
- `data/`: Directory containing data for Causal Emotion Entailment.

### Example Code Run
#### For Training
```bash
python run_cee.py
```
#### For Evaluation
```bash 
python eval_cee.py
```



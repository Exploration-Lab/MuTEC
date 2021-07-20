## End to End Architecture for Causal Emotion Entailment
Install required libraries:
```bash
pip install -r requirements.txt
```
- `config.py`: Contains the hyperparameter configurations used. Change the hyperparameters and dataset locations in this file.
- `dataset.py`: Custom Dataset script.
- `model.py`: Contains the Task2 models implemented.
- `run_nli.py`: Run script.
- `eval_nli.py`: Eval Script.
- `utils.py`: Utility functions.
- `data/`: Directory containing data for Task2.

### Example Code Run
#### For Training
```bash
python run_nli.py
```
#### For Evaluation
```bash 
python eval_nli.py
```



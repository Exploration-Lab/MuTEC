## MuTEC (CSE): Cause Span Extraction 
Install required libraries:
```bash
pip install -r requirements.txt
```

- `beam.py`: Contains code for BeamStart and BeamEnd.
- `config.py`: Contains the hyperparameter configurations used. Also contains the dataset locations.
- `dataset.py`: Custom Dataset script.
- `e2e_model.py`: Contains the MuTEC(CSE) model implementation.
- `run_cse.py`: Run script.
- `eval_cse.py`: Eval Script.
- `features_utils`: Utility functions for creating features for span prediction task.
- `utils_e2e.py`: Utility functions.
- `data/`: Directory containing data for MuTEC (CSE).

### Example Code Run
#### For Training
```bash
python run_cse.py
```
#### For Evaluation
```bash 
python eval_cse.py
```



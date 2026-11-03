# MachineLearning1-Assignment3
## k-Nearest Neighbours on the Wine Dataset

Implementation and evaluation of a k-nearest neighbours classifier using the Wine dataset. The classifier computes Euclidean distances explicitly rather than relying on a prebuilt k-NN estimator.

### Experiments

- Compare neighbourhood sizes `1, 2, 3, 5, 10, 15, 20, 25, 50`.
- Evaluate a multiclass classification task.
- Run one-versus-rest experiments for each class over repeated random splits.
- Display confusion matrices, error rates, accuracy, precision, recall and F1 scores.
- Summarise metrics using averages, medians, standard deviations and percentile intervals.

### Setup and execution

```bash
git clone https://github.com/MattiaTinfena/MachineLearning1-Assignment3.git
cd MachineLearning1-Assignment3
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy pandas matplotlib scikit-learn
python main.py
```

The main script loads Wine through scikit-learn. Edit `kval`, `reps` and the split fraction in `main.py` to configure the experiments.

### Source organisation

- `knn.py`: distance calculation, neighbour selection and voting.
- `loadData.py`: normalisation and train/test splitting.
- `functions.py`: metrics summaries and plots.
- `main.py`: experiment workflow.

### Evaluation notes

The current loader normalises the complete dataset before splitting. For an evaluation that prevents preprocessing leakage, fit the normalisation parameters on the training partition only. Random splits and vote ties can also affect the results.

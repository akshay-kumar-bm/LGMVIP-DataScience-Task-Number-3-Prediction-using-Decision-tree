# Iris Species Prediction with a Decision Tree

A short LetsGrowMore (LGMVIP) data-science internship task: classify Iris flowers into three species with a scikit-learn Decision Tree.

## What it does
Loads the classic Iris dataset (150 rows, 4 numeric features, 3 species), explores it, trains a `DecisionTreeClassifier`, visualises the tree and evaluates it on a held-out split.

## Approach (from `iris_.ipynb`)
1. Load `Iris.csv`; inspect with `head`, `info`, `describe`; confirm no null values.
2. Encode `Species` to a numeric `Target` (Iris-setosa=0, Iris-versicolor=1, Iris-virginica=2); drop `Id`.
3. Correlation heatmap and boxplots (seaborn).
4. Train/test split: 67/33, `random_state=42`.
5. Fit `DecisionTreeClassifier()` with default parameters; plot the tree with `sklearn.tree.plot_tree`.
6. Evaluate with a classification report and confusion matrix.

## Results shown in the notebook
Test set of 50 samples: setosa 19/19 correct, versicolor 15/15 correct, virginica 15/16 correct (one virginica predicted as versicolor), i.e. 49/50 correct (about 98%). Note: this is a single small split, with no cross-validation.

## Tech stack
Python, pandas, NumPy, seaborn, matplotlib, scikit-learn, Jupyter.

## Project structure
```
Iris.csv                       # dataset
iris_.ipynb                    # analysis + model
iris.pdf                       # exported notebook
Recording 2024-04-10 121254.mp4  # screen recording of the task
```

## Run
```bash
pip install pandas numpy seaborn matplotlib scikit-learn jupyter
jupyter notebook iris_.ipynb
```

## Limitations
Beginner exercise: default hyper-parameters, one split, no tuning or cross-validation, no requirements file.

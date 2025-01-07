# Support Vector Machine Project: Dual Quadratic Programming and MVP Decomposition

## Project Overview
This project focuses on implementing Support Vector Machine (SVM) models for classification tasks using the MNIST dataset. Key tasks include:

1. **SVM Dual Quadratic Problem**: Solves the dual optimization problem using convex programming techniques.
2. **MVP Decomposition**: Implements the Most Violating Pair (MVP) decomposition method for subproblem optimization.
3. **Multiclass Classification**: Extends SVM to handle three-class problems using one-vs-all or one-vs-one strategies.

The implementation is part of the course **Optimization Methods for Machine Learning** in the Master's program.

---

## Dataset

### Description
The MNIST dataset consists of handwritten digit images with corresponding labels. This project focuses on the following labels:
- **Binary classification tasks**: Labels 1 and 5.
- **Multiclass classification**: Labels 1, 5, and 7.

### Preprocessing Steps
- Selected 1000 samples per label.
- Normalized feature values to the range [0, 1] using `MinMaxScaler`.
- Converted binary labels to `+1` and `-1` as required by SVM.
- Split the dataset into 80% training and 20% testing subsets.

---

## Methods and Implementation

### Libraries Used
- **NumPy**: Numerical computations and array operations.
- **Pandas**: Data manipulation.
- **scikit-learn**: Preprocessing, metrics, and utilities.
- **CVXOPT**: Solving quadratic programming problems.
- **time**: Measuring execution time.

### Components

#### 1. **SVM Dual Quadratic Problem**
- **Kernel**: Polynomial kernel defined as \\((1 + \\langle x, y \\rangle)^\\gamma\\) with \\(\\gamma = 3\\).
- **Optimization**: Uses `cvxopt.solvers.qp` to solve the quadratic programming problem for dual variables (\\(\\alpha\\)).
- **Hyperparameter Tuning**: Grid search on \\(C\\) and \\(\\gamma\\).
- **Outputs**:
  - Training/testing accuracy, objective function values, and KKT condition violations.

#### 2. **MVP Decomposition**
- **Working Set Selection**: Implements MVP to focus on subsets of variables.
- **Kernel Optimization**: Updates subproblems using reduced quadratic programming.
- **Hyperparameter Tuning**: Evaluates \\(q\\) (dimension of subproblems) to optimize efficiency.
- **Outputs**:
  - Training/testing accuracy, iteration count, and execution time.

#### 3. **Multiclass Classification**
- **Strategy**: One-vs-All or One-vs-One.
- **Kernel**: Polynomial kernel.
- **Outputs**:
  - Confusion matrix and classification metrics (accuracy, precision, recall).

---

## Results

| Task                      | \\(C\\) | \\(\\gamma\\) | \\(q\\) | Training Accuracy | Testing Accuracy | Iterations | KKT Violations | Time  |
|---------------------------|-------|-------------|-------|-------------------|------------------|------------|----------------|-------|
| SVM Dual Quadratic Problem | 1     | 3           | -     | 98.81%            | 97.25%           | 100        | 1520           | 7.43s |
| MVP Decomposition          | 1     | 3           | 16    | 91.63%            | 89.91%           | 404        | 1124           | 9.07s |
| MVP Decomposition          | 1     | 3           | 2     | 88.69%            | 86.76%           | 1000       | 876            | 6.28s |

### Observations
- **Accuracy**: The dual quadratic programming approach achieves the best accuracy.
- **Efficiency**: The MVP decomposition method is faster.
- **Overfitting**: Polynomial kernels with \\(\\gamma = 3\\) prevent overfitting, while RBF kernels result in overfitting.

---

## Execution Instructions

1. **Data Extraction**:
   Run `Project_2_dataExtraction.py` to load and preprocess the MNIST dataset. Not that the data file is too big to be uploaded here so all the data is not uploaded on Github

2. **Binary Classification**:
   - **Dual Quadratic Problem**: Run `run_1_Mushtaque.py`.
   - **MVP Decomposition**: Run `run_2_Mushtaque.py`.

3. **Multiclass Classification**:
   Execute `run_3_Mushtaque.py` for the three-class problem.

---

## Conclusions
- **Best Accuracy**: SVM Dual Quadratic Problem (97.25% testing accuracy).
- **Fastest Execution**: MVP Decomposition with \\(q = 2\\) (6.28s).
- **Multiclass Feasibility**: Demonstrates potential for handling complex classification tasks.

---

## License
This project is part of an academic assignment for the course **Optimization Methods for Machine Learning**.

# Missing Values Project

## Data

- **XOR Data**:  
    XOR datasets are generated using the "exclusive or" (XOR) logical operation. Each data point typically has two binary input features. The output label is 1 if exactly one input is 1 (i.e., the inputs differ), and 0 otherwise. XOR datasets are a classic example in machine learning because they are not linearly separable, making them useful for testing algorithm capabilities.

- **OpenML Dataset**:  
    OpenML datasets are sourced from [OpenML](https://www.openml.org/), an online platform for sharing and organizing machine learning datasets. These datasets are widely used for benchmarking, reproducibility, and collaborative research. In this project, OpenML datasets provide real-world scenarios for evaluating missing value handling techniques. The datasets can be easily fetch using Sklearn's implementation [`fetch_openml`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_openml.html)

## Missing Values Generation

Three types of missing values are considered:

- **MCAR (Missing Completely At Random)**:  
    The probability of a value being missing is unrelated to any observed or unobserved data; missingness is entirely random.  
    *Example*: A survey response is lost due to a technical glitch.

- **MAR (Missing At Random)**:  
    The probability of missingness is related to observed data, but not to the missing data itself.  
    *Example*: Older participants are less likely to answer a question, but among people of the same age, missingness is random.

- **MNAR (Missing Not At Random)**:  
    The probability of missingness is related to the unobserved (missing) data itself.  
    *Example*: People with higher incomes are less likely to report their income.

For this generation we will rely on [Jenga](https://github.com/schelterlabs/jenga), a python library optimized to for benchmarking data inputation models

## Models for benchmarking:

- **Zero Imputation**:  
    Missing values are replaced with zeros. This simple approach is fast and easy to implement, but it may introduce bias if zero is not a representative value for the feature. Implementation is straightforward using sklearn's [SimpleImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html#sklearn.impute.SimpleImputer).

- **Mean Imputation**:  
    Missing values are filled with the mean of the observed values for that feature. This method preserves the overall mean but can underestimate variability and may not be suitable for skewed distributions. Implementation is straightforward using sklearn's [SimpleImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html#sklearn.impute.SimpleImputer).

- **KNN Imputation**:
    The K-Nearest Neighbors (KNN) algorithm is used to impute missing values based on the values of the k most similar instances (neighbors). This approach can capture local data structure but may be computationally intensive for large datasets. Scikit-learn provides an implementation: [KNNImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html#sklearn.impute.KNNImputer)

- **MICE (IterativeImputer)**:  
    Multiple Imputation by Chained Equations (MICE), implemented as `IterativeImputer` in scikit-learn, models each feature with missing values as a function of the other features in a round-robin fashion. This iterative process can produce more accurate imputations by leveraging relationships between features. Scikit-learn provides an implementation: [IterativeImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html#sklearn.impute.IterativeImputer)

- **PROMISSING**:  
    Implementation of the [PROMISSING: Pruning Missing Values in Neural Networks](https://arxiv.org/abs/2206.01640) model, adapted to PyTorch in `promising.py`. This approach enables neural networks to handle missing values by pruning the corresponding neurons, allowing for robust inference without imputation.

- **mPROMISSING**:  
    A modified version of the PROMISSING model, also based on the [PROMISSING: Pruning Missing Values in Neural Networks](https://arxiv.org/abs/2206.01640) paper and implemented in PyTorch within `promising.py`. This variant introduces additional modifications or enhancements to the original PROMISSING method for improved handling of missing data.

Aditional Useful links: 
- [Sklearn's Explanation of Imputation of missing values](https://scikit-learn.org/stable/modules/impute.html) 
- [JENGA - A Framework toStudy the Impact of Data Errors on the Predictions of Machine Learning Models](https://openproceedings.org/2021/conf/edbt/p134.pdf)

## Larry's Propsed Implementation
### Proposed Name
COMPASS-Net (COMbinatorial PAttern Sub-modelS Network)
Note: This name is roberto's proposal, to be approved by Larry

### Problem Addressed
Robust inference in fully-connected, feed-forward neural networks (FC-FF NNs) when 1 or 2 input features are missing at test time, without resorting to imputation.

### Key Idea
Train a separate sub-network for every missing-feature pattern of size 1 or 2. At inference time, each sample is routed to the sub-network that matches its mask of missing features.

### Number of Sub-networks
For an input dimension $n$:
- **No-feature-missing model** 1
- **1-feature-missing patterns:** $n$
- **2-features-missing patterns:** $C(n, 2) = \frac{n(n-1)}{2}$

**Total:** 1 + $n + \frac{n(n-1)}{2} = \frac{n(n+1)}{2}$ sub-models.

*(Extension to larger masks is possible.)*

### Architecture of Each Sub-network
- **Hidden layers:** Identical depth, width, activations, and initial hyper-parameters as the original “full” model.
- **Input layer:** Width = $n - k$, where $k \in \{1, 2\}$.
- **Output layer:** Unchanged.

### Training Data per Sub-network
Use the same full training set, but delete the corresponding feature(s) in every row so that the network sees a consistent mask during training. No synthetic imputation is performed.

### Training Procedure
1. Start from the architecture/hyper-parameters of a well-performing baseline FC-FF NN.
2. For each mask pattern:
    - Delete the masked feature column(s) from every row.
    - Re-initialize and train the model end-to-end (optimizer, epochs, loss, etc. unchanged from the baseline).
3. Optionally train the baseline “no-features-missing” model alongside the masked sub-networks.

### On-chip Deployment
All sub-networks are stored simultaneously on the target inference chip. The hardware automatically selects and runs the correct sub-network once it detects which feature(s) are absent in an input vector.

### Inference Workflow
1. Detect which features (if any) are missing in the incoming sample.
2. Remove those feature slots.
3. Dispatch the reduced vector to the sub-network whose mask matches.

### Evaluation Protocol
- Create perturbed copies of the hold-out/validation set by randomly dropping 1 feature in $X\%$ of rows and 2 features in $Y\%$ of rows (e.g., $X = 3\%$, $Y = 1\%$).
- For each row, run the appropriate sub-network.
- Report pattern-wise and overall accuracy; sweep several $(X, Y)$ pairs for robustness.

### Scalability / Extensions
- If domain knowledge or deployment data indicate more than 2 missing features are likely, extend the ensemble to larger mask sizes (combinatorial growth is the only constraint).
- Can incorporate model-compression or parameter-sharing techniques to stay within chip memory if $n$ is large.



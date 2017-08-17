# naive_bayes
This is a python implementation of Naive Bayes Classifier.
This implementation is now supporting:
```
	- Multivariate Bernoulli Model
	- Multinomial Model
	- Transformed Weight-normalized Complement Naive Bayes Model (TWCNB)
```

If you want to know more about TWCNB, see the following paper:
```
	Tackling the Poor Assumptions of Naive Bayes Text Classifiers,
	Rennie+, ICML'03
	http://machinelearning.wustl.edu/mlpapers/paper_files/icml2003_RennieSTK03.pdf
```

Finally, you can use some feature selection methods as follows:

* feature selection using word frequency (FS_FREQ)
* feature selection using PMI (FS_PMI)

# Requirements

```
pip -r requirements.txt
```

# Install

```
cd ./calc_score
python setup.py build_ext --inplace 
```

# Usage
## Training
```python
from .naive_bayes import (
	MultivariateBernoulli as MB
	Multinomial as MNMAL
	TWCNB
)
from .const import (
	NBParam,
	LEARNING_MLE,
	LEARNING_MAP,
	FS_NO,
	FS_FREQ,
	FS_PMI
)

# hyper parameters
SNUM = 100  # # of stopwords
VSIZE = 10000  # # of vocabulary
alpha = 2  # hyper parameter for dirichlet distribution for MAP

param = NBParam(
	LEARNING_MAP,  # choose MLE or MAP
	FS_FREQ,  # choose method for feature selection (NO, FREQ or PMI)
	SNUM,
	VSIZE,
	alpha
)

# init model (MB, MNMAL or TWCNB)
model = TWCNB(param, use_cython=True)

# get training data
"""
The format for training data is following:
	{class1: [[word_111, word112, ..., word_cij, ..., word_11N], ...]
	 class2: [[word_211, word212, ..., word_cij, ..., word_21N], ...]
	 ...
	}
where word_cij is j-th word for document i of class c.

If you want to use MB, bag of words for docs should be unique.
"""
train = make_data()

# training
model.learn(train)
model.save("/path/to/save")

```

## Testing
```python
from .naive_bayes import (
	MultivariateBernoulli as MB
	Multinomial as MNMAL
	TWCNB
)

# load model (MB, MNMAL or TWCNB)
model = TWCNB.load("/path/to/load")

# get test data
"""
The format for test data is follwing:
	[
	 [word_11, word12, ..., word_ij, ..., word_1N],
	 [word_21, word22, ..., word_ij, ..., word_2N],
	 ...
	]
where word_ij is j-th word for test document j.
"""
test = make_data()

# testing
labels = model.predict(test)
```

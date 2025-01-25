[Lecture link](https://youtu.be/111E7iaRgY4?list=PL8PYTP1V4I8D4BeyjwWczukWq9d8PNyZp)

When we classify, we try do:
1. Binary classification (Two classes)
2. Multi-class classification (Multiple classes)
3. Structured prediction (Exponential / infinite labels to predict from)

There are two types of predictions:
1. Unconditioned --> Predict the probability of a single variable P(X)
2. Conditioned --> Predict the probability of an output given an input P(Y|X)
		* Sequence to one
		* Sequence to sequence

Types of Unconditioned predictions: -- [Image link](https://drive.google.com/file/d/1Avhy_xL425AXbvthtYwdu1DSyz1Dmq5R/view?usp=sharing)

1. Left to Right / Autoregressive prediction
	1. We use RNNs/LSTMs/Transformer LMs to model them
2. Left to Right / Markov Chain (order n-1)
	1.  Instead of choosing everything from the past, we select a range and we predict based on that.
	2. n-gram LM, feed-forward LM etc.
3. Independent Prediction
	1. Predict one token at a time and calculate the probability
	2. Does not work as good
	3. unigram models.
4. Bi-direction Prediction
	1. Predicting the token based on what's before and what's after.
	2. Masked language models (BERT etc.)
	3. Disadvantage is that they won't be able to create a new sentence.

Types of Conditioned Predictions  -- [Image Link](https://drive.google.com/file/d/1WFssme30LRuWg66p9WSrTG5seOgpjkmB/view?usp=drive_link)
1. Autoregressive Conditioned Predictions
	1. Seq2Seq models
2. Non-autoregressive Conditioned Predictions
	1. sequence labeling, non-autoregressive MT
	2. For use cases like POS tagging

Smoothing Methods:
* Goodman: An empirical study of smoothing techniques for language modelling.


LM Evaluation:
1. Log-likelihood
	1. Likelihood of the parameters given the data
	2. This is aligned towards an available ground truth
	3. Mostly used during the training phase against test.

2. Per-word Log Likelihood:
	1.  We'll divide the Log-likelihood by the number of tokens
	2. Given that not tokenisers are all the same, if two models are compared, we'll want to use the same value for the denominator, accounting for the log-likelihood, else it would become an unequal comparison.
3. Entropy - Per-word (Cross Entropy)
	1. Almost the same as the above except that we take log2 here.
	2. Because probability theory and information theory are connected.
	3. 




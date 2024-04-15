1. Slow computation because of long sequences
2. Vanishing and Exploding gradients
	* When the computational graphs are created, the deeper the computation for back propagation / chain rule is, difficult it is for the floating points to be stored precisely leading to numbers that are closer to zero or very big numbers.
	* Especially because the each time step is solely depend on the computations from the past.
3. It is difficult to access information form the past. Since each layer is dependent on the output from the past, it gets difficult to keep the track of all the past per se. The past gets condensed as a single input to the next layer.
4. Transformers solve this by:
	* There is an encoder part and a decoder part in Transformer
	* For all the words/tokens the model is trained on, each of the vocabulary is associated with a unique input ID. For each input ID there is a corresponding Embedding vector. 
		* The dimension of the embedding vector is referred to as d_model in the code
	* Positional embedding is a fixed vector that matches the size of the embedding vector. It is only computed once and is reused for every sentence during the training and for the inference
	* Adding the  Embedding vector and the  Positional embedding will result in the same sized vector for each token in the sentence
		![[Pasted image 20240415131009.png]]
		![[Pasted image 20240415131220.png]]
	* 
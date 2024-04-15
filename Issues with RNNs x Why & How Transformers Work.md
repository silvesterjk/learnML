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
	* Self Attention / Multi Headed Attention is paying attention to words/tokens that are associated/related.
		* Self attention as the tokens in the sentence pay attention to each other.
		![[Pasted image 20240415132625.png]]
		![[Pasted image 20240415132715.png]]
			![[Pasted image 20240415133501.png]]
	* If we do not want some values to interact with each other, we can artificially set the values to negative infinity, which can then become 0 when softmax is applied. 
	* Since words against each other are meeting in the diagonal of the matrix, it tends to always have high score. 
		* Multi-head Attention:
			![[Pasted image 20240415133754.png]]
	
	* The positional embedded input is made 4 copies:
		
		* First is sent to Query Q (seq x d_model) -> Multiplied by a parametric matrix W_Q (d_model x d_model) = Q' (seq x d_model) --> Split into number of pieces as many as d_model dimensions (owing to the number of heads). Each of it is called a d_k.
		
		* Second is sent to the Key K (seq x d_model) -> Multiplied by a parametric matrix W_K (d_model x d_model) = K' (seq x d_model) --> Split into number of pieces as many as d_model dimensions (owing to the number of heads). Each of it is called a d_k.
		
		* Third is sent to the Value V (seq x d_model) --> Multiplied by a parametric matrix W_V (d_model x d_model) = V' (seq x d_model) --> Split into number of pieces as many as d_model dimensions (owing to the number of heads). Each of it is called a d_k.
		
		![[Pasted image 20240415174910.png]]

		* Query, Key and Values --> Query and Key are analogous associating noun with adjective. Values are associated with Keys.
			![[Pasted image 20240415175226.png]]
		* Fourth is sent to the Residual connection --> Original positional embedded data in the loop.
	* Layer Normalisation: 
		* Normalises each layer with its own mean and variance
	* In Encoder-Decoder style Transformer: Keys and Values come from the Encoder block and the query comes from the decoder block of the masked multi headed attention
		* In masked multi headed attention, the upper triangular matrix is masked with negative infinity. 
		
	* During Training: Input (seq x d_model) will output the same dimension (seq x d_model). This output is sent to the decoder which also take the input and outputs in same dimension (seq x d_model). This goes through a linear layer and that goes through softmax layer, which outputs the predicted probability score for the entire corpus.
		* Cross entropy loss is used to backpropagate and tune the parameters. 

	During inference: The output goes to the linear layer from the decoder (seq x d_model) is transformed to (seq x vocab_size), which is passed through the softmax layer that also outputs (seq x vocab_size) which predicts the token Ti. Ti is now appended to the current input text and sent to the mode once again as the new decoder input. This process repeats until the model outputs EOS token.
	* Since the output for the previous steps have already be computed, we multiply the previous matrix with the embedding vector for the newly generated word and passed as the new input.
	* Greedy --> Take the best softmax score at all times to produce the next text.
	* Beam Search --> At each step, select the top B words and evaluate all the possible next words for each of them and at each step keeping the top B most probably sequences. 
	
	Advantage:
	* All of this is done in one time step unlike RNNs or LSTMs.
	* Parallelizable through GPUs as the K, Q, V weights are the same for a particular head. 
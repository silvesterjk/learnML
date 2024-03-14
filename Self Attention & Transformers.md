
Attention:
* The main idea of attention is to add a bunch of new paths from the encoder to the decoder one per input value so that each step of the decoder can directly access input values.
* There is no one way to add attention to an encoder-decoder model. There are only conventions and some popular methods. 
* Objective is to find how similar the outputs are from the encoder to the outputs from the decoder. A similarity score essentially.
* In the case of cosine similarity, the numerator calculates the similarity and the denominator scales the value between -1 and 1. 
	![[Pasted image 20240314132928.png]]
* The most common way to calculate the similarity for attention is to just calculate the numerator for the cosine similarity. This is because the denominator simply scales the magnitude of the similarity score to be between negative 1 and 1. 
	* Since we are using the same number of inputs from the encoder and decoder, the magnitude of the similarity need not be scaled. 
	* This we can stick to just the numerator or simply the dot product. 
* Similarity scores are passed through softmax to get the weightage of the percent of input to be used
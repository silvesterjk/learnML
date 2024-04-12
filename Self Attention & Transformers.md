
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


---------------------------------

Transformers

* Word embedding is a technique to convert words into numbers.
- Word embedding assigns numerical values to words based on their meaning.
- The same word embedding network is used for each word or symbol. That gives us the flexibility to work with sentences of varied length, as the word embedding network's weights remain constant.
- Backpropagation optimizes the weights in word embedding.
- Positional coding helps keep track of word order in transformers. Positional coding allows a transformer to keep track of word order and track relationships between words.
	- Position values for each word come from the corresponding y-axis coordinates in the squiggles.
	- Positional coding is added to word embeddings to create a unique sequence of positional values for each word.
	- Transformers use self-attention mechanism to correctly associate words and calculate similarity between words.

Transformer self-attention calculates similarity scores between words
- Similarity scores are computed using dot product between query and key values
- Softmax function is applied to preserve order and normalize similarity scores

Self-attention calculates values for each word based on their similarity to other words
- Values are calculated by scaling representation of the word and adding them
- Queries, keys, and values are calculated simultaneously for each word
- Self-attention helps establish word context and relationships in sentences and paragraphs

The transformer model consists of an encoder and a decoder
- The encoder encodes the input phrase into numerical representations
- The decoder decodes the encoded input phrase into Spanish

Encoder-decoder attention helps the transformer keep track of the relationships between input and output sentences.
- Encoder-decoder attention allows the decoder to keep track of significant words in the input.
- Encoder-decoder attention involves calculating similarities between the EOS token in the decoder and each word in the encoder.
- Softmax function determines the percentage of each input word to be used in determining the first translated word.
- Encoder-decoder attention values are calculated by scaling the values of each input word based on softmax percentages.
- Residual connections allow the encoder-decoder attention to focus on the relationships between output and input words.
- A fully connected layer is used to select one of the four output tokens.

Transformers use word embedding, positional coding, self-attention, encoder-decoder attention, and residual connections to translate phrases.
- Word embedding converts words into numbers and positional encoding keeps track of word order.
- Self-attention tracks word relationships in input and output sentences.
- Encoder-decoder attention ensures important words in the input are not lost in translation.
- Residual connections allow each subunit to focus on solving one part of the problem.

--------------------------------- 

In essense:

* We tokenise the input sentences (corpus)
* We create an embedding -- turn the tokens into the vector representations.
* Positional encoding -- add additional vectors to the word embedding to retain the word order and semantic meaning of the sentence (corpus). 
*  
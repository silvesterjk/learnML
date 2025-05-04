
Attention:
* The main idea of attention is to add a bunch of new paths from the encoder to the decoder one per input value so that each step of the decoder can directly access input values.
* There is no one way to add attention to an encoder-decoder model. There are only conventions and some popular methods. 
* Objective is to find how similar the outputs are from the encoder to the outputs from the decoder. A similarity score essentially.
* In the case of cosine similarity, the numerator calculates the similarity and the denominator scales the value between -1 and 1.
* The most common way to calculate the similarity for attention is to just calculate the numerator for the cosine similarity. This is because the denominator simply scales the magnitude of the similarity score to be between negative 1 and 1. 
	* Since we are using the same number of inputs from the encoder and decoder, the magnitude of the similarity need not be scaled. 
	* This we can stick to just the numerator or simply the dot product. 
* Similarity scores are passed through softmax to get the weightage of the percent of input to be used.
* q is the query and k is the key

Attention Score functions:
* 1 Using multi-layer perceptron
		Challenge is that taking dot product is not possible.
* 2 Bilinear
		This is the one that's used in transformers (with a small tweak).
* 3 Dot product
		This requires the q and k to be of the same size.
		This is mainly limited by expecting the query and key in the same space.
* 4. Scaled Dot Product
		* The scale of the dot product increases as the dimensions get larger.
		* We can scale by the size of the vector.
		* We do this by dividing the q^Tk with the square root of the key vector 
		* The scale of the input values to the softmax affects the peakiness of the output.
		* Dividing the attention scores by the square root of the dimension of the key vector is a normalisation technique employed in self-attention. This step helps in controlling the magnitude of the attention scores and stabilising the learning process.
		* Consider a scenario where the dimensionality of the key vector is large. The dot product between the query and key vectors can become disproportionately large due to the increased dimensionality. This can lead to issues during training, such as exploding gradients or vanishing gradients, which hinder the learning process and make it difficult for the model to effectively capture dependencies between tokens.
		* To mitigate this problem, the attention scores are divided by the square root of the dimension of the key vector. By doing so, we scale down the attention scores, ensuring that their magnitude remains manageable. This scaling prevents the dot product from becoming too large and helps stabilize the learning process.

Types of Transformers:
* Encoder-Decoder (T5, MBART)
* Decoder-only (GPT, Llama)

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

**In essense:**
* We tokenise the input sentences (corpus)
* We create an embedding -- turn the tokens into the vector representations.
* Positional encoding -- add additional vectors to the word embedding to retain the word order and semantic meaning of the sentence (corpus). 
*  Which is then sent to the attention layer, which is then added with the residual connection then normalised before it is sent to the FNN. The output of the FNN is also added and normalised with the residual connection.
* The logits from the FNN that are added and normalised are then sent as the the input to the multi headed attention layer of the decoder block, and this attention layer takes other input from the masked-multi headed attention layer of the decoder block.
* Output of the multi headed attention layer of the decoder block is then sent to another FNN which is then added with its residual connection and then normalised. 
* Which goes through a linear layer and then a softmax later to output predicted probabilities against the text corpus. 

--------------------------------- 

# Newer Adaptations
## Linear Attention Transformers

One approach is transforming traditional attention mechanisms to be linear rather than quadratic in complexity. Researchers have shown that self-attention can be expressed as "a linear dot-product of kernel feature maps" which reduces complexity from O(N²) to O(N) and allows for an iterative implementation that connects transformers to RNNs. [ArXiv](https://arxiv.org/abs/2006.16236) This approach makes autoregressive transformers dramatically faster for very long sequences.

## RWKV (Receptance Weighted Key Value)

Perhaps the most successful RNN adaptation for parallelization is RWKV. RWKV "combines the efficient parallelizable training of transformers with the efficient inference of RNNs" through a linear attention mechanism that allows the model to be formulated as either a transformer or an RNN. [ArXiv](https://arxiv.org/abs/2305.13048) RWKV can be:

- Parallelized during training (like a transformer)
- Run sequentially during inference (like an RNN)
- Scale to large model sizes (up to 14 billion parameters)

RWKV is parallelizable because the time-decay of each channel is data-independent and trainable. Instead of adjusting time-decay within channels as in traditional RNNs, RWKV moves information between channels with different decay rates. [GitHub](https://github.com/BlinkDL/RWKV-LM)

## Teacher Forcing for Parallel Training

A technique called "Teacher Forcing" allows decoder components to be parallelized during training by using ground truth outputs instead of the model's previous outputs. This makes the RNN-style components trainable in parallel while maintaining sequential generation during inference.

## RetNet (Retention Network)

Another recent approach following RWKV's success is RetNet, which uses a retention mechanism rather than attention. RetNet "combines the parallel representation and the recurrent representation to achieve a balance between efficiency and performance" and can generate multi-scale sequence representations.

## Hybrid Transformer-RNN Models

Some approaches directly combine transformer and RNN components:

One approach uses attention layer to create memories associated with context patterns rather than the inputs themselves. This relieves pressure on the RNN structure's weights and allows the model memory to focus on remembering the input.
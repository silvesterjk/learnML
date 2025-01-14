
* BERT is a encoder only model
* The text is split into tokens
* Which is then passed on to an embedding layer that embeds the information of a token in relation to other token in a multidimensional space as each token are mapped to n dimensional vectors say 512 for example.
	* We usually use cosine similarity to understand the similarity of different words/tokens.
* Which is given a positional embedding to retain the sequence information.
	* These vectors are learned for the positions and not for the input token itself
	* For example first token of any input input sequence will receive the first positional embedded vector. 
		![[Pasted image 20240508114405.png]]![[Pasted image 20240508114248.png]]
		![[Pasted image 20240508114348.png]]
	* Causal Mask: Minus infinity that goes through the softmax layer would output 0 wherever the mask is applied. Which is typically implemented in autoregressive fashion. ![[Pasted image 20240508114609.png]]
*  How BERT uses left and right context?
		![[Pasted image 20240508120709.png]]
		![[Pasted image 20240508120851.png]]
		We use segment embeddings to differentiate tokens that belong to segment A and segment B
		| CLS token captures information that captures information from all other tokens:
		![[Pasted image 20240508121500.png]]
		| Text Classification : In this context we input say 14 tokens and the system will output 14 tokens. The first token is passed on to a linear layer. Which then classifies the input into one of the predefined categories. 
		![[Pasted image 20240508122025.png]]
		| Question Answering: ![[Pasted image 20240508122314.png]]
		![[Pasted image 20240508122319.png]]
		| Here the context and the question is differentiated using the SEP token.
		![[Pasted image 20240508122545.png]]
		**Extractive vs. Abstractive QA:** There are two main types of question answering with BERT:
		| Extractive QA: Here, BERT identifies the answer within the provided context (passage). The answer length depends on the actual answer in the text, not a fixed number of tokens (words).
		| Abstractive QA: This is less common and involves BERT generating a new answer based on its understanding of the context. Even here, the output length is based on the information needed to convey the answer, not a set number.
		 BERT considers the entire context (passage) to answer the question. 
		 ![[Pasted image 20240508123550.png]]
		

* Language AI, a field characterised by the development of systems capable of understanding and generating human language.
* Language AI encompass things that are beyond LLMs. Retrieval systems can boost the capabilities of LLMs can be considered as one.
* Bag-of-words was one of the first techniques. A method to represent unstructured text. Which starts with tokenisation, the process of splitting up the text into words/subwords. 
	* Then create an unique set of tokens to form vocabulary. Total number of unique tokens then will become the vocabulary size.
	* Using vocabulary, we then check how often a word in each sentence appears and creates a bag of words that keeps the count of words.
* Given that the Bag-of-words does not take the semantics of the language into account, word2vec was introduced in 2013. This is one of the successful attempt at understanding the meaning of the text via embeddings. Which learns the semantic representations of words by training on vast amounts of textual data. Say, the whole of Wikipedia. 
	* They used neural networks to do that.
	* Using these neural networks, word2vec generates word embeddings by looking at which other words they tend to appear next to in a given sentence.
	* If the two words tend to have the same neighbours, their embeddings will be closer to one another and vice versa.
	* Since the size of embeddings is fixed, their properties (dimensions) are chosen to create a mental representation of the word.
* There are many types of embeddings, like word embeddings and sentence embeddings that are used to indicate different levels of abstractions (word versus sentence).
	* Embeddings can be created for different types of inputs too. Such as document, sentence, word, tokens and so on.
* Attention selectively determines which words are most important in a given sentence.
* Self-attention can attend to different positions within a single sequence, thereby more easily and accurately representing the input sequence.
* BERT approaches by adopting a technique called _masked language modeling_ and this method masks a part of the input for the model to predict.
	* The `[CLS]` token in BERT (and models that followed its architecture like RoBERTa, DistilBERT, etc.) serves a crucial purpose: **to provide a unified representation for the entire input sequence, specifically for classification and sentence-level tasks.**
		* **Input Sentence:** Let's say your input sentence is: "**This food is delicious!**"
		* **Tokenization:** Break the sentence into word pieces that BERT understands (using BERT's tokeniser). It might become something like: `["this", "food", "is", "delicious", "!"]`
		* **Tokenization:** Break the sentence into word pieces that BERT understands (using BERT's tokeniser). It might become something like: `["this", "food", "is", "delicious", "!"]`
		* Once fed into BERT: BERT will process this and give you output vectors for _every token_, **including the `[CLS]` token.**
		* Take the `[CLS]` token's output vector. This vector is BERT's representation of the _entire sentence_. 
		* **Simple Classifier:** Feed this `[CLS]` vector into a very simple classifier – often just **a single linear layer**. This layer learns to map BERT's `[CLS]` representation to the sentiment categories (positive or negative).
			* Imagine a linear layer that takes the `[CLS]` vector (which is usually quite long, like 768 dimensions in base BERT) and reduces it to 2 numbers. These two numbers represent the "score" for "positive" and "negative" sentiment.
		* **Prediction:** Apply a **softmax** function to the output of the linear layer. This converts the scores into probabilities. The category with the higher probability is your predicted sentiment.
	* In simpler terms: Imagine you want a summary of a whole paragraph. The `[CLS]` token is like asking BERT to create a special "summary token" that represents the meaning of the entire input sentence or sequence. Because it's trained within the Transformer architecture using self-attention, it learns to effectively aggregate information from all the words in the sentence. You then just need to look at the output representation of this single `[CLS]` token to get your sentence-level understanding.
	* BERT-like models are commonly used for _transfer learning,_ which involves first pretraining it for language modeling and then fine-tuning it for a specific task. For instance, by training BERT on the entirety of Wikipedia, it learns to understand the semantic and contextual nature of text. Then, we can use that pretrained model to _fine-tune_ it for a specific task, like text classification.
* Encoder-only models such as BERT can be classified as _representation models_. Similarly decoder-only can be called _generative models_.
* Without much change in the architecture, _representation models_ mainly focus on representing language, for instance, by creating embeddings, and typically do not generate text. In contrast, _generative models_ focus primarily on generating text and typically are not trained to generate embeddings.

* Attention is All You Need: 
	* Compared to the recurrence network, the Transformer could be trained in parallel, which tremendously sped up training.
	* 


________________________________________________________________________

Thoughts from the topic -- Will be added to [[Thoughts for Research]] :
1. Non playable characters (NPCs) have often been referred to as AI even though many are mere if-else statements. Huh! Interesting. 
2. Can we encode alphabets with tokens instead of 0 and 1s?
3. Can we use a local language encoder to boost the performance of the model in languages the models are not trained on?
4. Can we do dimensionality reduction with the decoder-only models?
5. Can we add variations to the model output post inference? --> One with greedy search (Temp: 0) and the further steps are built on the previous output for preference tuning.
6. Lego: Can we create a classifier (+ context embeddings) to identify the programming language and use that as a gating mechanism to choose between different specialists? -- See how single task executions such as in BERT-like models can be useful here.
	1. I suspect if abstraction can help a lot in make the models more parameter efficient during inference. Wonder if RL can come in handy here too.
7. 
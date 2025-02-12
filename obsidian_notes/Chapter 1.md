
* Language AI, a field characterised by the development of systems capable of understanding and generating human language.
* Language AI encompass things that are beyond LLMs. Retrieval systems can boost the capabilities of LLMs can be considered as one.
* Bag-of-words was one of the first techniques. A method to represent unstructured text. Which starts with tokenisation, the process of splitting up the text into words/subwords. 
	* Then create an unique set of tokens to form vocabulary. Total number of unique tokens then will become the vocabulary size.
	* Using vocabulary, we then check how often a word in each sentence appears and creates a bag of words that keeps the count of words.
* Given that the Bag-of-words does not take the semantics of the language into account, word2vec was introduced in 2013. This is one of the successful attempt at understanding the meaning of the text via embeddings. Which learns the semantic representations of words by training on vast amounts of textual data. Say, the whole of Wikipedia. 
	* They used neural networks to do that.
	* Using these neural networks, word2vec generates word embeddings by looking at which other words they tend to appear next to in a given sentence.
	* If the two words tend to have the same neighbours, their embeddings will be closer to one another and vice versa.




Thoughts from the topic -- Will be added to [[Thoughts for Research]] :
1. Non playable characters (NPCs) have often been referred to as AI even though many are mere if-else statements. Huh! Interesting. 
2. Can we encode alphabets with tokens instead of 0 and 1s?
3. Can we use a local language encoder to boost the performance of the model in languages the models are not trained on?
4. Can we do dimensionality reduction with the decoder-only models?

 What is different about the LLaMA model from a Transformers model?
 * Transformer vs LLaMA! ![[Pasted image 20240507224644.png]]
 * Review of FNN: ![[Pasted image 20240507215324.png]]
 * RMSNorm is used against the Layer Normalisation as we only the invariance to obtain the layer normalisation and not we can do it without re-centering as it is was done with the vanilla method.
 * Rotary positional encoding deals with two tokens at a time, since the attention captures the `intensity` of how two words are related to each other, this method tells the attention mechanism the distance between the two tokens involved. So given two tokens, we create a vector that represents their distance. 
	 * It is also important to note that the RoPE shows long-term decay with the growth of relative distance. Which means that the `itensity` of relationships between the tokens encoded in RoPE will be numerically smaller as the distance between them grows. 
	 * This is a desirable property. 
 * KV caching is used to avoid unnecessary computation on tokens that the model has already seen during inference. 
 * Attention: ![[Pasted image 20240507222752.png]]
 * During Inference one: ![[Pasted image 20240507222843.png]]
 * During the next inference:![[Pasted image 20240507222903.png]]
	and so on.
 * We can notice that we are recomputing the same things over and again, if we can cache it then it turns out to be convenient. ![[Pasted image 20240507223017.png]]
 * Step 1 with KV Cache:![[Pasted image 20240507223100.png]]
 * Step 2 with KV cache:![[Pasted image 20240507223117.png]]
	 and so on.
 * KV cache introduces a new bottle neck and it increased the speed in the GPU operations, it now has to move around tokens for the computation which the GPUs struggle at (data transfer).  
	 * Multi-query attention with KV cache solves this problem. By removing the h dimension from the K and the V, while retaining it for Q. Which means that all the different query heads will share the same keys and values.
	 * The quality only degrades a little bit but the performance gain is very high. 
 * LLaMA uses Grouped Multi-Query attention instead.![[Pasted image 20240507224145.png]]

* LoRA freezes the pretrained weights (d x k)
* We then create two matrices B (d x r) and A (r x k)
* We multiply B and A to get a d x k dimension matrix
* Which is then added with the pretrained weights
* During the training process, we backpropagate loses only to the newly created weights![[Pasted image 20240508154610.png]]
* And the benefits are that we have less parameters to train and faster backpropagation. ![[Pasted image 20240508154815.png]]
* Rank is the number of vectors that are linear independent of each other
* Task adaptation in LLMs are low rank in nature, thus, if we can pack a lot of linearly independent vectors and for a dense matrix with task specific information, the hypothesis is that the updates to the weights also have low "intrinsic rank" during adaptation. 
	* This is commonly used in compression algorithms
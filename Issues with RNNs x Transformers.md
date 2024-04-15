1. Slow computation because of long sequences
2. Vanishing and Exploding gradients
	* When the computational graphs are created, the deeper the computation for back propagation / chain rule is, difficult it is for the floating points to be stored precisely leading to numbers that are closer to zero or very big numbers.
	* Especially because the each time step is solely depend on the computations from the past.
3. It is difficult to access information form the past. Since each layer is dependent on the output from the past, it gets difficult to keep the track of all the past per se. The past gets condensed as a single input to the next layer.
4. Transformers solve this by:
	* There is an encoder part and a decoder part in Transformer
	* 
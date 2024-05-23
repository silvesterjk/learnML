* Networks are complex against deep learning. Here the data is of arbitrary size and with complex topology. There is no notion of up and down, or left and right. Like in matrices, tensors or arrays. Here a node can be connected to arbitrary number of nodes, including itself. As they are dynamic, they have multimodal features. 
* Via representational learning, we can automatically learn features from the graph.
	* Map nodes to d dimensional embeddings such that similar nodes in the network are embedded close together.
* Applications of Graph ML
	* We can define tasks at: Node level, subgraph level and edge level
	* Node: Predict a property of a node
	* Link Prediction: Predict missing links between two nodes
	* Graph classification: Classification of different molecules as different graphs
	* Clustering: Grouping relevant nodes together
	* Graph generation: Drug discovery
	* Graph evolution: Physical simulations
* Strongly connected components (SCCs) can be identified but not every node is part of a nontrivial strongly connected component. 
	* In Component -- Nodes that reach the SCC
	* Out Component -- Nodes that can be reached from the SCC

[[Knowledge Graph]]

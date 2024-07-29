<details>  
<summary> Networks vs Natural Graphs?</summary> 
  -- Social Network where we are connected with the people we know and the people they know etc.
  -- Communications and Transactions where we make phone calls, financial transactions etc.
  -- Biomedicine where the interactions between genes/proteins regulate life
  -- How our thoughts are captured by the hidden connections between the billions of neurons.
  -- Information with relational structure:
	  -- Information/knowledge: Once is generally linked to another
	  -- Software
	  -- Similarity Networks: Connecting similar data points 
	  -- Relational Structures: Molecules, scene graphs, 3D shapes, physics-particle simulation etc.
  Note: Sometimes the distinction between a network and a graph is blurred. 
</details> 

<details>  
<summary>Why are networks complex?</summary> 
  -- In comparison to text or images, which has spacial locality such as a pixel to the left or to the right, word that came before or the word that comes after, networks can be of arbitrary size and topological structure unlike in grids or arrays.
	  -- While modern deep learning toolbox is designed for simple sequences and grids
  -- There cannot be any reference points and no fixed node ordering.
  -- They are often dynamic and have multi modal features.
  
</details> 

<details>  
<summary>What is representational learning?</summary> 
  -- Here the goal is to learn the function f that takes the node u and maps it to the d dimensional real valued vectors.
	  -- f:u --> R^d -- To learn a neural network
  -- Here the similar nodes in the nodes are embedded close together.
  
</details> 

<details>  
<summary>What are the applications are graphs?</summary> 
  -- There are many tasks attached to graphs such as graph level prediction and generation
  -- The same can be scaled down to community or sub graph, node-level and edges-level
  -- Applications in areas such as protein folding where predicting the protein's 3D structure solely based on its amino acid sequence can/do have a huge impact in drug discovery etc.
	  -- Spatial Graph: Where Nodes are the amino acids in the protein sequence and edges are the proximity between amino acids (residues)
  -- PinSage: Graph-based recommender
	  -- In the case of a Pinterest board, if we're trying to find another similar image, via graph where similar things are embedded together is found to be doing this task much better than the image only based recommendation.
  -- Drug Side effects
	  -- Given then drugs attack certain proteins, we can check if drug A and B are given to someone, what are the chances, side effect A would occur.
	  -- Through protein-protein and protein-drug interaction.
  -- Drug Discovery
	  -- By connecting atoms as nodes with its chemical bonds as edges, we predict the pool of candidates that are promising candidates for antibiotic drugs for example.
		  -- A GNN graph classification model can predict that
		  -- Antibiotics are small molecular graphs
	  -- Graph generation can lead to discoveries of new and novel molecules where the use cases include predicting molecules with high drug likeliness and optimise existing molecules to have desirable properties.
  -- Traffic prediction
	  -- Road segments are nodes and connectivity between them are edges.
	  -- This can help us predict the ETA
  --  Physics Simulation
	  -- A graph evolution task can take break down of things to smaller chunks and predict the velocity of those chunks' movements --> to predict the new position of those chunks.
		  -- In reliably simulating the world.
-- Weather Forecasting
	-- Graphcast by Deepmind creates a multi-mesh message passing on the atmosphere to rollout a forecast of the weather.
</details> 

<details>  
<summary>How to define a graph? </summary> 
  -- A graph's design influences how it represents information
  -- A graph with same number of nodes and edges, with similar connections can be used to map several things at once. Such as: 1. Say 4 actors in 4 nodes and what movies they appear in, in edges 2. Friendship network 3. Protein-protein network. etc.
  -- Directed and Undirected Graphs. Directed where there is a source and a destination (arcs) and Undirected where we can represent relationships, collaboration (symmetrical and reciprocal)
  --  In Undirected cases --> Node Degree: The number of edges adjacent to a node i. Average degree turns out to be 2E/N
  -- Bipartite Graph: Where we can create two groups of nodes and only these groups are connected with each other and not within. Such as authors-to-papers network and actors-to-movies networks etc.
  -- Folded networks: If two authors worked on a paper, these two on one group would have at least one connection on the other side, which will help us to map co-authorship.
  -- We can represent a graph as a adjacency matix:
	  A_ij = 1 if there is a link from node i to node j
	  A_io = 0 if otherwise
  -- Hence the undirected graphs are naturally symmetric and the directed graphs are not
  -- 
  --  
  
</details> 
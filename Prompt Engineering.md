https://www.promptingguide.ai/

LLM Settings: 

* Temperature --> Control randomness
* Top P --> High value would enable the model to look at more possible words
* Max Length --> Set the maximum length for the response
* Frequency penalty --> Avoids repeating the same words/tokens
* Presence penalty --> To avoid repeating certain phrases

Elements of a Prompt: A prompt contains any of the following elements:

* Instruction - a specific task or instruction you want the model to perform
* Context - external information or additional context that can steer the model to better responses
* Input Data - the input or question that we are interested to find a response for
* Output Indicator - the type or format of the output.

Prompt Design:
* Start simple
	* Be specific, simple and concise. 
* Iterate over functionality overtime.

Techniques:

* Zero-shot prompting:
		`Classify the text into neutral, negative or positive. Text: I think the vacation is okay. Sentiment:
		`Response: Neutral
* Few-Shot Prompting:
		`A "whatpu" is a small, furry animal native to Tanzania. An example of a sentence that uses the word whatpu is: We were traveling in Africa and we saw these very cute whatpus. To do a "farduddle" means to jump up and down really fast. An example of a sentence that uses the word farduddle is:`
		`Response:  When we won the game, we all started to farduddle in celebration.`
		_
		_
		Another example with multiple prompts would be:
		`This is awesome! // NegativeThis is bad! // PositiveWow that movie was rad! // PositiveWhat a horrible show! //`
		`Output: Negative`
* Chain-of-Thought (CoT) Prompting
	* Zero-shot COT Prompting
		`I went to the market and bought 10 apples. I gave 2 apples to the neighbor and 2 to the repairman. I then went and bought 5 more apples and ate 1. How many apples did I remain with?Let's think step by step.
		
		`OUTPUT:`
		`First, you started with 10 apples. 
		`You gave away 2 apples to the neighbor and 2 to the repairman, so you had 6 apples left.
		`Then you bought 5 more apples, so now you had 11 apples.
		`Finally, you ate 1 apple, so you would remain with 10 apples.` 
		
	* Automatic Chain-of-Thought (Auto-CoT)
		![[Pasted image 20240505121050.png]]
		`Auto-CoT consists of two main stages:
		`Stage 1): Question clustering: Partition questions of a given dataset into a few clusters.
		`Stage 2): Demonstration sampling: Select a representative question from each cluster and generate its reasoning chain using Zero-Shot-CoT with simple heuristics.
		
	* Self-Consistency
	* Generated Knowledge Prompting
		![[Pasted image 20240505121638.png]]
		`Input: Greece is larger than Mexico.
		`Knowledge: Greece is approximately 131,957 sq km, while Mexico is approximately 1,964,375 sq km, making Mexico 1,389% larger than Greece.
		
		`Input: Glasses always fog up.
		`Knowledge: Condensation occurs on eyeglass lenses when water vapor from your sweat, breath, and ambient humidity lands on a cold surface, cools, and then changes into tiny drops of liquid, forming a film that you see as fog. Your lenses will be relatively cool compared to your breath, especially when the outside air is cold.
		
		`Input: A fish is capable of thinking.
		`Knowledge: Fish are more intelligent than they appear. In many areas, such as memory, their cognitive powers match or exceed those of ’higher’ vertebrates including non-human primates. Fish’s long-term memories help them keep track of complex social relationships.
		
		`Input: A common effect of smoking lots of cigarettes in one’s lifetime is a higher than normal chance of getting lung cancer.
		`Knowledge: Those who consistently averaged less than one cigarette per day over their lifetime had nine times the risk of dying from lung cancer than never smokers. Among people who smoked between one and 10 cigarettes per day, the risk of dying from lung cancer was nearly 12 times higher than that of never smokers.
		
		`Input: A rock is the same size as a pebble.
		`Knowledge: A pebble is a clast of rock with a particle size of 4 to 64 millimetres based on the Udden-Wentworth scale of sedimentology. Pebbles are generally considered larger than granules (2 to 4 millimetres diameter) and smaller than cobbles (64 to 256 millimetres diameter).
		
		`Input: Part of golf is trying to get a higher point total than others.
		`Knowledge:

		`Output:`
		
		`The objective of golf is to play a set of holes in the least number of strokes. A round of golf typically consists of 18 holes. Each hole is played once in the round on a standard golf course. Each stroke is counted as one point, and the total number of strokes is used to determine the winner of the game.`

		`Golf is a precision club-and-ball sport in which competing players (or golfers) use many types of clubs to hit balls into a series of holes on a course using the fewest number of strokes. The goal is to complete the course with the lowest score, which is calculated by adding up the total number of strokes taken on each hole. The player with the lowest score wins the game.

		The next step is to integrate the knowledge and get a prediction. I reformatted the question into QA format to guide the answer format.

		Confidence is very high:
		`Question: Part of golf is trying to get a higher point total than others. Yes or No?
		`Knowledge: The objective of golf is to play a set of holes in the least number of strokes. A round of golf typically consists of 18 holes. Each hole is played once in the round on a standard golf course. Each stroke is counted as one point, and the total number of strokes is used to determine the winner of the game. 
		`Explain and Answer:`

		`No, the objective of golf is not to get a higher point total than others. Rather, the objective is to play a set of holes in the least number of strokes. The total number of strokes is used to determine the winner of the game, not the total number of points.`

		Confidence is a lot lower:
		`Question: Part of golf is trying to get a higher point total than others. Yes or No?
		`Knowledge: Golf is a precision club-and-ball sport in which competing players (or golfers) use many types of clubs to hit balls into a series of holes on a course using the fewest number of strokes. The goal is to complete the course with the lowest score, which is calculated by adding up the total number of strokes taken on each hole. The player with the lowest score wins the game. 
		`Explain and Answer:`

		`Yes, part of golf is trying to get a higher point total than others. Each player tries to complete the course with the lowest score, which is calculated by adding up the total number of strokes taken on each hole. The player with the lowest score wins the game.
	
	* Prompt Chaining
		`The first prompt below extracts the relevant quotes from the document given the question. Note that for simplicity, we have added a placeholder for the document `{{document}}`. To test the prompt you can copy and paste an article from Wikipedia such as this page for [prompt engineering(opens in a new tab)](https://en.wikipedia.org/wiki/Prompt_engineering). Due to larger context used for this task, we are using the `gpt-4-turbo` model from OpenAI. You can use the prompt with other long-context LLMs like Claude.`

		PROMPT 1:
		`You are a helpful assistant. Your task is to help answer a question given in a document. The first step is to extract quotes relevant to the question from the document, delimited by ####. Please output the list of quotes using <quotes></quotes>. Respond with "No relevant quotes found!" if no relevant quotes were found.
		`####
		`{{document}}
		`####`

		![[Pasted image 20240505123310.png]]

		OUTPUT:
		`<quotes>
		`- Chain-of-thought (CoT) prompting[27]
		`- Generated knowledge prompting[37]
		`- Least-to-most prompting[38]
		`- Self-consistency decoding[39]
		`- Complexity-based prompting[41]
		`- Self-refine[42]
		`- Tree-of-thought prompting[43]
		`- Maieutic prompting[45]
		`- Directional-stimulus prompting[46]
		`- Textual inversion and embeddings[59]
		`- Using gradient descent to search for prompts[61][62][63][64]
		`- Prompt injection[65][66][67]
		`</quotes>`

		PROMPT 2: 
		`Given a set of relevant quotes (delimited by <quotes></quotes>) extracted from a document and the original document (delimited by ####), please compose an answer to the question. Ensure that the answer is accurate, has a friendly tone, and sounds helpful.####{{document}}####<quotes>- Chain-of-thought (CoT) prompting[27]- Generated knowledge prompting[37]- Least-to-most prompting[38]- Self-consistency decoding[39]- Complexity-based prompting[41]- Self-refine[42]- Tree-of-thought prompting[43]- Maieutic prompting[45]- Directional-stimulus prompting[46]- Textual inversion and embeddings[59]- Using gradient descent to search for prompts[61][62][63][64]- Prompt injection[65][66][67]</quotes>

		OUTPUT:
		`The prompting techniques mentioned in the document include:
		`1. Chain-of-thought (CoT) prompting[27]
		`2. Generated knowledge prompting[37]
		`3. Least-to-most prompting[38]
		`4. Self-consistency decoding[39]
		`5. Complexity-based prompting[41]
		`6. Self-refine[42]
		`7. Tree-of-thought prompting[43]
		`8. Maieutic prompting[45]
		`9. Directional-stimulus prompting[46]
		`10. Textual inversion and embeddings[59]
		`11. Using gradient descent to search for prompts[61][62][63][64]
		`12. Prompt injection[65][66][67]
		
		`Each of these techniques employs unique strategies to enhance or specify the interactions with large language models to produce the desired outcomes.
	* Tree of Thoughts (ToT)
		![[Pasted image 20240505123744.png]]
	* RAG:
		![[Pasted image 20240505125707.png]]
	* Automatic Reasoning and Tool-use (ART)
		![[Pasted image 20240505130056.png]]
	* Automatic Prompt Engineer (APE)
		![[Pasted image 20240505130117.png]]
		`The first step involves a large language model (as an inference model) that is given output demonstrations to generate instruction candidates for a task. These candidate solutions will guide the search procedure. The instructions are executed using a target model, and then the most appropriate instruction is selected based on computed evaluation scores.

	* Active-Prompt:
		![[Pasted image 20240505131216.png]]
		`Above is an illustration of the approach. The first step is to query the LLM with or without a few CoT examples. _k_ possible answers are generated for a set of training questions. An uncertainty metric is calculated based on the _k_ answers (disagreement used). The most uncertain questions are selected for annotation by humans. The new annotated exemplars are then used to infer each question.`
	
	* Directional Stimulus Prompting
		![[Pasted image 20240505131751.png]]
	* PAL (Program-Aided Language Models
		![[Pasted image 20240505132404.png]]
		`It differs from chain-of-thought prompting in that instead of using free-form text to obtain solution it offloads the solution step to a programmatic runtime such as a Python interpreter.`
	* ReAct Prompting
		
	* 

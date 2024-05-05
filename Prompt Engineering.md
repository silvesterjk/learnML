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

* 
import ollama
import json

class PromptAssemblyAndAnswerGeneration:
    def __init__(self, model_name="llama3"):
        self.model_name = model_name

    def generate_answer(self, prompt):
        try:
            # Use ollama's chat function to get the model's response
            response = ollama.chat(model=self.model_name, messages=[{"role": "user", "content": prompt}])
            return response['message']['content']
        except Exception as e:
            print(f"Error generating answer: {e}")
            return None

    def build_prompt(self, query, ranked_segments):
        # Step 1: Create the prompt by concatenating the ranked segments

        assembled_prompt = "\n".join([
            f"[Source: {seg.get('source', 'Unknown')}, Page: {seg.get('page', 'Unknown')}, Line: {seg.get('line_no', 'Unknown')}] {seg['content']}"
            for seg in ranked_segments
        ])
        
        # Step 2: Build the full prompt for answer generation

        prompt = f"""
        The user wants to learn about: "{query}" and wants responses ranked by how well they provide the correct explanation, definition, procedure, etc.
        Do not go beyond what is given as input.
        Include the source citation after each line or claim based on the content segments. Cite them like [Source: pdf, Page: 12, Line: 21 ] or [Source: mp3, Start: 12, End: 21 ]. Each source can be reused across multiple lines if needed, but stay faithful to which source supports which point.
        Here are the top-ranked content segments:
        {assembled_prompt}
        Based on the above content, provide a concise and comprehensive answer, citing sources inline.
        """
        return prompt


class SemanticAxisReranker:
    def __init__(self, model_name="llama3"):
        self.model_name = model_name
        self.answer_generator = PromptAssemblyAndAnswerGeneration(model_name)

    def build_prompt(self, query, axis, segments):

        numbered_segments = "\n".join([
            f"{i+1}. \"{seg.get('content', '')}\"" for i, seg in enumerate(segments)
        ])

        

        prompt = f"""
        You are an expert assistant. The user wants to learn about: "{query}" and wants responses ranked by how well they provide a {axis}.
        Here are some content segments:
        {numbered_segments}
        Score each segment from 0 to 1 based on how well it aligns with the axis: {axis}. Respond only with a list of scores like: [0.9, 0.2, 0.75, ...]
        """

        return prompt

    def rerank(self, query, axis, segments):
        prompt = self.build_prompt(query, axis, segments)

        try:
            response = ollama.chat(model=self.model_name, messages=[{"role": "user", "content": prompt}])
            output = response['message']['content']
            scores_str = output.split("[")[-1].split("]")[0]
            scores = [float(s.strip()) for s in scores_str.split(",")]

            for i, seg in enumerate(segments):
                seg['axis_score'] = scores[i] if i < len(scores) else 0.0
            

            return sorted(segments, key=lambda x: x['axis_score'], reverse=True)

        except Exception as e:
            print("Error in reranking:", e)
            for seg in segments:
                seg['axis_score'] = 0.0
            return segments

    def generate_answer(self, query, axis, segments):
        # Step 1: Rerank the segments based on relevance to the axis
        ranked_segments = self.rerank(query, axis, segments)
        
        # Step 2: Build the prompt by concatenating the ranked segments
        prompt = self.answer_generator.build_prompt(query, ranked_segments)

        # Step 3: Send the prompt to the LLM for final answer generation
        final_answer = self.answer_generator.generate_answer(prompt)

        return final_answer

"""
# Example Usage
if __name__ == "__main__":
    reranker = SemanticAxisReranker(model_name="llama3:latest")

    # Define segments with some example content
    segments = [
        {"content": "Recursion is a technique where a function calls itself.", "source": "PDF", "axis_score": 0.8},
        {"content": "The professor said recursion is useful for solving problems with repetitive structure.", "source": "Audio @ 00:23", "axis_score": 0.9},
        {"content": "This figure shows the stack of recursive calls in a depth-first search.", "source": "Diagram", "axis_score": 0.75}
    ]

    # Generate the final answer
    final_answer = reranker.generate_answer("What is recursion?", "explanation", segments)

    # Print the final answer
    print(final_answer)
"""

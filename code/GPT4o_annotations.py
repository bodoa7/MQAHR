import json
import os
from openai import OpenAI
from tqdm import tqdm
import time

class MQAHRAnnotator: 
    
    def __init__(self, api_key: str = None):
        
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o"
    
    def get_prompt(self, question: str, options: str, gold_answer: str, reasoning: str) -> str:
   
        return f"""You are a hallucination analysis expert specializing in medical reasoning evaluation.

Task: Analyze the reasoning process to identify the first hallucination type and difficulty level.

Hallucination Types:
H1 – Input Misunderstanding: (Easy: misses clear condition | Medium: misinterprets context | Hard: misreads complex conditions)
H2 – Knowledge Confusion: (Easy: mixes basic facts | Medium: confuses related facts | Hard: misrecalls rare knowledge)
H3 – Knowledge Fabrication: (Easy: fabricates entity | Medium: blends true/false | Hard: constructs fictional mechanism)
H4 – Reasoning Bias: (Easy: single logical jump | Medium: faulty causal link | Hard: fully incorrect reasoning)

Question: {question}
Options: {options}
Gold Answer: {gold_answer}
Reasoning Process: {reasoning}
"""
    
    def annotate_sample(self, sample: dict) -> dict:
      
        try:
            
            prompt = self.get_prompt(
                question=sample['question'],
                options=sample['options'],
                gold_answer=sample['gold_answer'],
                reasoning=sample['model_think_process']
            )
            
          
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert medical reasoning evaluator. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
          
            result = json.loads(response.choices[0].message.content)
       
            sample['hallucination_type'] = result['hallucination_type']
            sample['first_step'] = result['first_step']
            sample['difficulty'] = result['difficulty']
            
            return sample
            
        except Exception as e:
            print(f"Error annotating sample {sample.get('id', 'unknown')}: {e}")
            return None
    
    def annotate_dataset(self, input_file: str, output_file: str):
       
        print(f"Loading data from {input_file}...")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Total samples: {len(data)}")
        
   
        annotated = []
        failed = []
        
        for sample in tqdm(data, desc="Annotating"):
            result = self.annotate_sample(sample)
            
            if result:
                annotated.append(result)
            else:
                failed.append(sample['id'])
            
      
            if len(annotated) % 100 == 0:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(annotated, f, ensure_ascii=False, indent=2)
            
            time.sleep(0.1) 
        

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(annotated, f, ensure_ascii=False, indent=2)
        
        print(f"\nAnnotation complete!")
        print(f"Success: {len(annotated)}/{len(data)}")
        print(f"Failed: {len(failed)}")
        if failed:
            print(f"Failed IDs: {failed}")



if __name__ == "__main__":

    annotator = MQAHRAnnotator(api_key="your-openai-api-key")
    
    annotator.annotate_dataset(
        input_file='data.json',
        output_file='annotated_data.json'
    )
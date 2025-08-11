import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class TextSummarizer:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass it directly.")
        
        self.client = OpenAI(api_key=self.api_key)
    
    def chunk_text(self, text, max_chunk_size=3000):
        """Split text into chunks for processing"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            if current_size + len(word) + 1 > max_chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_size = len(word)
            else:
                current_chunk.append(word)
                current_size += len(word) + 1
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def summarize_chunk(self, chunk, summary_type="detailed"):
        """Summarize a single chunk of text"""
        prompts = {
            "brief": "Provide a brief 2-3 sentence summary of the following text:",
            "detailed": "Provide a comprehensive summary of the following text, highlighting key points and main ideas:",
            "bullet": "Create a bullet-point summary of the key points from the following text:"
        }
        
        prompt = prompts.get(summary_type, prompts["detailed"])
        
        try:
            # Use max_tokens for gpt-3.5-turbo (older model)
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates clear and concise summaries."},
                    {"role": "user", "content": f"{prompt}\n\n{chunk}"}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            raise Exception(f"Error summarizing text: {str(e)}")
    
    def summarize(self, text, summary_type="detailed"):
        """Summarize text, handling long texts by chunking"""
        if len(text) < 3000:
            return self.summarize_chunk(text, summary_type)
        
        # For longer texts, chunk and summarize each chunk
        chunks = self.chunk_text(text)
        chunk_summaries = []
        
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}...")
            summary = self.summarize_chunk(chunk, summary_type)
            chunk_summaries.append(summary)
        
        # If we have multiple chunks, create a final summary
        if len(chunk_summaries) > 1:
            combined_text = '\n\n'.join(chunk_summaries)
            final_summary = self.summarize_chunk(
                combined_text, 
                summary_type
            )
            return final_summary
        
        return chunk_summaries[0]
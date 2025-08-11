import os
import requests
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class EnhancedSummarizer:
    """Enhanced summarizer that supports multiple AI providers"""
    
    def __init__(self):
        self.providers = {
            'openai': self._init_openai(),
            'perplexity': self._init_perplexity(),
            'deepseek': self._init_deepseek(),
            'gemini': self._init_gemini(),
            'grok': self._init_grok(),
            'claude': self._init_claude()
        }
        
        # Filter out None providers (missing API keys)
        self.providers = {k: v for k, v in self.providers.items() if v is not None}
        
        if not self.providers:
            raise ValueError("No AI providers available. Please set at least one API key.")
        
        print(f"[INFO] Available AI providers: {list(self.providers.keys())}")
    
    def _init_openai(self):
        """Initialize OpenAI client"""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            return None
        try:
            return {
                'client': OpenAI(api_key=api_key),
                'models': [
                    'gpt-4o',
                    'gpt-4o-mini', 
                    'gpt-4-turbo',
                    'gpt-4',
                    'gpt-3.5-turbo',
                    'o1-preview',
                    'o1-mini'
                ],
                'type': 'openai'
            }
        except Exception as e:
            print(f"[WARNING] OpenAI initialization failed: {e}")
            return None
    
    def _init_perplexity(self):
        """Initialize Perplexity API"""
        api_key = os.getenv('PERPLEXITY_API_KEY')
        if not api_key:
            return None
        return {
            'api_key': api_key,
            'base_url': 'https://api.perplexity.ai/chat/completions',
            'models': [
                'sonar-pro',
                'sonar-reasoner',
                'sonar-deep-research',
                'llama-3.1-sonar-small-128k-online',
                'llama-3.1-sonar-large-128k-online',
                'llama-3.1-sonar-huge-128k-online'
            ],
            'type': 'perplexity'
        }
    
    def _init_deepseek(self):
        """Initialize DeepSeek API"""
        api_key = os.getenv('DEEPSEEK_API_KEY')
        if not api_key:
            return None
        return {
            'api_key': api_key,
            'base_url': 'https://api.deepseek.com/v1/chat/completions',
            'models': [
                'deepseek-reasoner',
                'deepseek-chat',
                'deepseek-v3',
                'deepseek-coder',
                'deepseek-math'
            ],
            'type': 'deepseek'
        }
    
    def _init_gemini(self):
        """Initialize Gemini API"""
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            return None
        return {
            'api_key': api_key,
            'base_url': 'https://generativelanguage.googleapis.com/v1beta/models',
            'models': [
                'gemini-2.5-pro',
                'gemini-2.5-flash', 
                'gemini-2.5-flash-lite',
                'gemini-2.0-flash',
                'gemini-1.5-pro',
                'gemini-1.5-flash',
                'gemini-exp-1121'
            ],
            'type': 'gemini'
        }
    
    def _init_grok(self):
        """Initialize Grok API"""
        api_key = os.getenv('XAI_API_KEY')
        if not api_key:
            return None
        return {
            'api_key': api_key,
            'base_url': 'https://api.x.ai/v1/chat/completions',
            'models': [
                'grok-3',
                'grok-3-mini-beta',
                'grok-3-reasoner',
                'grok-3-deepsearch',
                'grok-2-latest',
                'grok-2-vision-latest'
            ],
            'type': 'grok'
        }
    
    def _init_claude(self):
        """Initialize Claude/Anthropic API"""
        api_key = os.getenv('CLAUDE_API_KEY')
        if not api_key:
            print(f"[INFO] Claude API key not found")
            return None
        print(f"[INFO] Claude API key found, initializing...")
        return {
            'api_key': api_key,
            'base_url': 'https://api.anthropic.com/v1/messages',
            'models': [
                'claude-3-5-sonnet-20241022',
                'claude-3-5-sonnet-20240620',
                'claude-3-5-haiku-20241022',
                'claude-3-opus-20240229',
                'claude-3-sonnet-20240229',
                'claude-3-haiku-20240307'
            ],
            'type': 'claude'
        }
    
    def get_available_providers(self):
        """Get list of available providers with their models"""
        providers_info = {}
        
        for provider, config in self.providers.items():
            providers_info[provider] = {
                'models': config['models'],
                'type': config['type']
            }
            
            # Try to fetch live models from API (for providers that support it)
            try:
                live_models = self._fetch_live_models(provider, config)
                if live_models:
                    providers_info[provider]['models'] = live_models
            except Exception as e:
                # Use static models as fallback
                print(f"[INFO] Using static models for {provider}: {str(e)}")
        
        return providers_info
    
    def _fetch_live_models(self, provider, config):
        """Try to fetch live models from provider API"""
        try:
            if config['type'] == 'openai':
                # OpenAI models endpoint - get real models from official API
                client = config['client']
                models_response = client.models.list()
                # Filter for stable, widely available chat completion models
                chat_models = []
                stable_patterns = [
                    'gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo', 'gpt-4o', 
                    'o1-preview', 'o1-mini', 'chatgpt-4o'
                ]
                
                for model in models_response.data:
                    model_id = model.id
                    model_lower = model_id.lower()
                    
                    # Skip non-chat models
                    if any(skip in model_lower for skip in ['whisper', 'tts', 'dall', 'embedding']):
                        continue
                    
                    # Include stable model patterns
                    if any(pattern in model_lower for pattern in stable_patterns):
                        chat_models.append(model_id)
                
                # Sort by preference (newer/better models first)
                preferred_order = [
                    'gpt-4o', 'chatgpt-4o-latest', 'gpt-4-turbo', 'gpt-4', 
                    'o1-preview', 'o1-mini', 'gpt-3.5-turbo'
                ]
                
                def sort_key(model):
                    for i, preferred in enumerate(preferred_order):
                        if preferred in model.lower():
                            return i
                    return 999  # Put unknown models at the end
                
                chat_models.sort(key=sort_key)
                return chat_models[:12] if chat_models else config['models']  # Top 12 stable models
                
            elif config['type'] == 'perplexity':
                # Perplexity doesn't have a models endpoint, use static list
                return None
                
            elif config['type'] == 'deepseek':
                # DeepSeek models endpoint
                headers = {'Authorization': f'Bearer {config["api_key"]}'}
                response = requests.get('https://api.deepseek.com/v1/models', headers=headers)
                if response.status_code == 200:
                    models_data = response.json()
                    return [model['id'] for model in models_data.get('data', [])][:10]
                return None
                
            elif config['type'] == 'gemini':
                # Gemini models endpoint
                url = f"{config['base_url']}"
                params = {'key': config['api_key']}
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    models_data = response.json()
                    models = [model['name'].split('/')[-1] for model in models_data.get('models', []) 
                             if 'generateContent' in model.get('supportedGenerationMethods', [])][:10]
                    return models if models else None
                return None
                
            elif config['type'] == 'grok':
                # Grok models endpoint
                headers = {'Authorization': f'Bearer {config["api_key"]}'}
                response = requests.get('https://api.x.ai/v1/models', headers=headers)
                if response.status_code == 200:
                    models_data = response.json()
                    return [model['id'] for model in models_data.get('data', [])][:10]
                return None
                
        except Exception as e:
            print(f"[DEBUG] Live model fetch failed for {provider}: {str(e)}")
            return None
    
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
    
    def _get_system_prompt(self, summary_type):
        """Get system prompt based on summary type"""
        prompts = {
            "brief": "You are a helpful assistant that creates brief, concise summaries in 2-3 sentences.",
            "detailed": "You are a helpful assistant that creates comprehensive, detailed summaries highlighting key points and main ideas.",
            "bullet": "You are a helpful assistant that creates clear bullet-point summaries of key information.",
            "tutorial": "You are a professional tutorial-to-guide converter. Your task is to turn transcripts into clear, detailed, and self-contained written tutorials.",
            "professional": "You are a professional summarizer specializing in video and podcast transcripts. Your goal is to create clear, concise, and self-contained summaries.",
            "custom": "You are a helpful AI assistant that follows specific user instructions for content processing."
        }
        return prompts.get(summary_type, prompts["detailed"])
    
    def _get_user_prompt(self, text, summary_type, custom_prompt=None):
        """Get user prompt based on summary type"""
        if summary_type == "custom" and custom_prompt:
            return f"{custom_prompt}\n\nTranscript:\n{text}"
        
        prompts = {
            "brief": f"Provide a brief 2-3 sentence summary of the following YouTube video transcript:\n\n{text}",
            "detailed": f"Provide a comprehensive summary of the following YouTube video transcript, highlighting key points and main ideas:\n\n{text}",
            "bullet": f"Create a bullet-point summary of the key points from the following YouTube video transcript:\n\n{text}",
            "tutorial": f"""Turn the provided transcript into a clear, detailed, and self-contained written tutorial so that the reader can follow it without watching the video.

Instructions:
1. Preserve all technical steps exactly, including:
   - Commands, code snippets, file names, and settings
   - Menu paths and UI actions
   - Variable names, parameters, and values

2. Reorder for clarity if the transcript jumps around. Present steps in the correct logical sequence.

3. Explain briefly why each step is done, if the reasoning is given or can be inferred.

4. Avoid filler and small talk – focus on actionable content.

Output format:
- Title of the tutorial
- Short introduction (what is being built, prerequisites, outcome)
- Step-by-step instructions numbered in order. Each step:
  - Action description
  - Any required code/configuration
  - Expected result or verification tip
- Final result: summary of what should now be working

Tone: Clear, direct, and instructional. Assume the reader is following along while doing the task.

Transcript:
{text}""",
            "professional": f"""Create a clear, concise, and self-contained summary that captures all main points, context, and relevant details, so it can be understood without consuming the original content.

Instructions:
Structure the output as:

1. **Intro** – 1–2 sentences summarizing the overall topic and purpose.

2. **Key sections** in chronological order, each with:
   - A short title for the section
   - A concise paragraph (3–6 sentences) summarizing the core points

3. **Final takeaway** – 2–3 sentences summarizing the main insights or conclusions.

Keep essential context:
- Explain terms or events briefly if needed
- Retain numbers, names, and important facts exactly

Be selective:
- Cut filler, repetition, and small talk
- Focus on what is valuable for understanding the whole piece

Tone: Neutral, clear, and accessible.

Transcript:
{text}"""
        }
        return prompts.get(summary_type, prompts["detailed"])
    
    def _summarize_with_openai(self, provider_config, text, summary_type="detailed", model=None, custom_prompt=None):
        """Summarize using OpenAI"""
        try:
            client = provider_config['client']
            model = model or 'gpt-3.5-turbo'
            
            # Use more tokens for tutorial and professional summaries
            max_tokens = 500 if summary_type == "brief" else (1500 if summary_type in ["tutorial", "professional"] else 1000)
            
            # Determine if model needs max_completion_tokens vs max_tokens
            # Based on OpenAI API documentation and testing
            
            # Models that definitely need max_completion_tokens
            needs_completion_tokens = [
                'gpt-4o', 'gpt-4o-mini', 'chatgpt-4o-latest',
                'o1-preview', 'o1-mini',
                'gpt-4-turbo', 'gpt-4-turbo-2024-04-09', 'gpt-4-turbo-preview'
            ]
            
            use_completion_tokens = False
            
            # Check exact matches first
            if model in needs_completion_tokens:
                use_completion_tokens = True
            # Check for GPT-4 variants (most GPT-4 models need max_completion_tokens)
            elif model.startswith('gpt-4') and model not in ['gpt-4o', 'gpt-4o-mini']:
                # GPT-4 series generally needs max_completion_tokens
                use_completion_tokens = True
            # Check for o1 models
            elif model.startswith('o1-'):
                use_completion_tokens = True
            # Check for other newer model patterns
            elif any(pattern in model.lower() for pattern in ['4o', 'turbo-2024', 'latest']):
                use_completion_tokens = True
                
                
            
            
            
            # Check if model is an o1 model (different message format)
            is_o1_model = 'o1-' in model
            
            # Prepare messages based on model type
            if is_o1_model:
                # o1 models don't support system messages, combine system and user content
                combined_prompt = f"{self._get_system_prompt(summary_type)}\n\n{self._get_user_prompt(text, summary_type, custom_prompt)}"
                messages = [{"role": "user", "content": combined_prompt}]
            else:
                messages = [
                    {"role": "system", "content": self._get_system_prompt(summary_type)},
                    {"role": "user", "content": self._get_user_prompt(text, summary_type, custom_prompt)}
                ]
            
            # Prepare API call parameters
            api_params = {
                'model': model,
                'messages': messages
            }
            
            # o1 models don't support temperature parameter
            if not is_o1_model:
                api_params['temperature'] = 0.3
            
            # Use appropriate token parameter based on model
            if use_completion_tokens:
                api_params['max_completion_tokens'] = max_tokens
            else:
                api_params['max_tokens'] = max_tokens
            
            response = client.chat.completions.create(**api_params)
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise Exception(f"OpenAI error: {str(e)}")
    
    def _summarize_with_api(self, provider_config, text, summary_type="detailed", model=None, custom_prompt=None):
        """Generic method for API-based providers (Perplexity, DeepSeek, Grok)"""
        try:
            headers = {
                'Authorization': f'Bearer {provider_config["api_key"]}',
                'Content-Type': 'application/json'
            }
            
            # Choose default model based on provider
            if not model:
                if provider_config['type'] == 'perplexity':
                    model = 'sonar-pro'
                elif provider_config['type'] == 'deepseek':
                    model = 'deepseek-reasoner'
                elif provider_config['type'] == 'grok':
                    model = 'grok-3-mini-beta'
            
            # Use more tokens for tutorial and professional summaries
            max_tokens = 500 if summary_type == "brief" else (1500 if summary_type in ["tutorial", "professional"] else 1000)
            
            data = {
                'model': model,
                'messages': [
                    {"role": "system", "content": self._get_system_prompt(summary_type)},
                    {"role": "user", "content": self._get_user_prompt(text, summary_type, custom_prompt)}
                ],
                'max_tokens': max_tokens,
                'temperature': 0.3
            }
            
            response = requests.post(provider_config['base_url'], headers=headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"{provider_config['type'].title()} API error: {str(e)}")
        except KeyError as e:
            raise Exception(f"{provider_config['type'].title()} response format error: {str(e)}")
    
    def _summarize_with_gemini(self, provider_config, text, summary_type="detailed", model=None, custom_prompt=None):
        """Summarize using Gemini API"""
        try:
            model = model or 'gemini-2.5-flash'
            headers = {
                'Content-Type': 'application/json'
            }
            
            url = f"{provider_config['base_url']}/{model}:generateContent"
            params = {'key': provider_config['api_key']}
            
            # Use more tokens for tutorial and professional summaries
            max_tokens = 500 if summary_type == "brief" else (1500 if summary_type in ["tutorial", "professional"] else 1000)
            
            data = {
                'contents': [{
                    'parts': [{'text': self._get_user_prompt(text, summary_type, custom_prompt)}]
                }],
                'generationConfig': {
                    'maxOutputTokens': max_tokens,
                    'temperature': 0.3
                }
            }
            
            response = requests.post(url, headers=headers, params=params, json=data)
            response.raise_for_status()
            
            result = response.json()
            return result['candidates'][0]['content']['parts'][0]['text'].strip()
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Gemini API error: {str(e)}")
        except KeyError as e:
            raise Exception(f"Gemini response format error: {str(e)}")
    
    def _summarize_with_claude(self, provider_config, text, summary_type="detailed", model=None, custom_prompt=None):
        """Summarize using Claude/Anthropic API"""
        try:
            model = model or 'claude-3-5-sonnet-20241022'
            headers = {
                'Content-Type': 'application/json',
                'x-api-key': provider_config['api_key'],
                'anthropic-version': '2023-06-01'
            }
            
            # Use more tokens for tutorial and professional summaries
            max_tokens = 500 if summary_type == "brief" else (1500 if summary_type in ["tutorial", "professional"] else 1000)
            
            # Claude uses a different message format
            system_prompt = self._get_system_prompt(summary_type)
            user_prompt = self._get_user_prompt(text, summary_type, custom_prompt)
            
            data = {
                'model': model,
                'max_tokens': max_tokens,
                'temperature': 0.3,
                'system': system_prompt,
                'messages': [
                    {
                        'role': 'user',
                        'content': user_prompt
                    }
                ]
            }
            
            response = requests.post(provider_config['base_url'], headers=headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            return result['content'][0]['text'].strip()
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Claude API error: {str(e)}")
        except KeyError as e:
            raise Exception(f"Claude response format error: {str(e)}")
    
    def summarize_chunk(self, chunk, summary_type="detailed", provider="openai", model=None, custom_prompt=None):
        """Summarize a single chunk of text using specified provider"""
        if provider not in self.providers:
            # Fallback to first available provider
            provider = list(self.providers.keys())[0]
            print(f"[WARNING] Requested provider '{provider}' not available, using '{provider}'")
        
        provider_config = self.providers[provider]
        
        try:
            if provider_config['type'] == 'openai':
                return self._summarize_with_openai(provider_config, chunk, summary_type, model, custom_prompt)
            elif provider_config['type'] == 'gemini':
                return self._summarize_with_gemini(provider_config, chunk, summary_type, model, custom_prompt)
            elif provider_config['type'] == 'claude':
                return self._summarize_with_claude(provider_config, chunk, summary_type, model, custom_prompt)
            else:
                # Perplexity, DeepSeek, Grok
                return self._summarize_with_api(provider_config, chunk, summary_type, model, custom_prompt)
                
        except Exception as e:
            # Try fallback to OpenAI if available and not already using it
            if provider != 'openai' and 'openai' in self.providers:
                print(f"[WARNING] {provider} failed, falling back to OpenAI: {str(e)}")
                return self._summarize_with_openai(self.providers['openai'], chunk, summary_type, custom_prompt=custom_prompt)
            else:
                raise e
    
    def summarize(self, text, summary_type="detailed", provider="openai", model=None, custom_prompt=None):
        """Summarize text using specified provider, with chunking for long texts"""
        print(f"[INFO] Using {provider} for {summary_type} summary")
        
        if len(text) < 3000:
            return self.summarize_chunk(text, summary_type, provider, model, custom_prompt)
        
        # For longer texts, chunk and summarize each chunk
        chunks = self.chunk_text(text)
        chunk_summaries = []
        
        # Limit chunks for long texts to prevent timeouts  
        if len(chunks) > 5:
            print(f"[WARNING] Long transcript ({len(chunks)} chunks). Limiting to first 4 chunks to prevent timeout.")
            chunks = chunks[:4]
        
        for i, chunk in enumerate(chunks):
            try:
                print(f"[INFO] Processing chunk {i+1}/{len(chunks)} with {provider}...")
                summary = self.summarize_chunk(chunk, summary_type, provider, model, custom_prompt)
                chunk_summaries.append(summary)
            except Exception as e:
                print(f"[ERROR] Failed to process chunk {i+1}: {str(e)[:100]}...")
                # Continue with other chunks instead of failing completely
                continue
        
        if not chunk_summaries:
            raise Exception("Failed to process any chunks of the transcript")
        
        # If we have multiple chunks, create a final summary
        if len(chunk_summaries) > 1:
            combined_text = '\n\n'.join(chunk_summaries)
            print(f"[INFO] Creating final summary from {len(chunk_summaries)} processed chunks...")
            print(f"[DEBUG] Combined text length: {len(combined_text)} characters")
            try:
                print(f"[DEBUG] Starting final summary with {provider}...")
                final_summary = self.summarize_chunk(
                    combined_text, 
                    summary_type,
                    provider,
                    model,
                    custom_prompt
                )
                print(f"[SUCCESS] Final summary completed, length: {len(final_summary)} chars")
                return final_summary
            except Exception as e:
                print(f"[WARNING] Final summary failed, returning combined chunks: {str(e)[:100]}...")
                # Return the combined chunk summaries if final summary fails
                return '\n\n'.join(chunk_summaries)
        
        return chunk_summaries[0]


# Backward compatibility with existing TextSummarizer
class TextSummarizer(EnhancedSummarizer):
    """Backward compatible wrapper for existing code"""
    
    def __init__(self, api_key=None):
        # Set OpenAI API key if provided
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key
        
        super().__init__()
        
        # Ensure OpenAI is available for backward compatibility
        if 'openai' not in self.providers:
            raise ValueError("OpenAI API key is required for TextSummarizer compatibility.")
    
    def summarize(self, text, summary_type="detailed"):
        """Original interface - always uses OpenAI"""
        return super().summarize(text, summary_type, provider="openai")
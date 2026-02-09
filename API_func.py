# This file is to define the API calling functions for the agent, create a general function for each model 
# By default, we will use ask_model function to call the API. All test cases will be done using GPT-4o-mini.

import openai
from anthropic import Anthropic
from google import genai
from api_keys import openai_api_key, claude_api_key, gemini_api_key


def ask_gpt(question, model="gpt-4o", temperature=0.7, max_tokens=2000):
    """
    Ask a question to GPT models using OpenAI API.
    
    Args:
        question (str): The question to ask
        model (str): The GPT model to use (default: "gpt-4o")
        temperature (float): Sampling temperature (default: 0.7)
        max_tokens (int): Maximum tokens in response (default: 2000)
    
    Returns:
        str: The generated answer
    """
    openai.api_key = openai_api_key
    
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": question}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error calling GPT API: {str(e)}"


def ask_claude(question, model="claude-3-5-sonnet-20241022", temperature=0.7, max_tokens=2000):
    """
    Ask a question to Claude models using Anthropic API.
    
    Args:
        question (str): The question to ask
        model (str): The Claude model to use (default: "claude-3-5-sonnet-20241022")
        temperature (float): Sampling temperature (default: 0.7)
        max_tokens (int): Maximum tokens in response (default: 2000)
    
    Returns:
        str: The generated answer
    """
    try:
        client = Anthropic(api_key=claude_api_key)
        
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "user", "content": question}
            ]
        )
        return message.content[0].text
    except Exception as e:
        return f"Error calling Claude API: {str(e)}"


def ask_gemini(question, model="gemini-2.5-flash", temperature=0.7, max_tokens=2000):
    """
    Ask a question to Gemini models using Google Generative AI API.
    
    Args:
        question (str): The question to ask
        model (str): The Gemini model to use (default: "gemini-2.5-flash")
        temperature (float): Sampling temperature (default: 0.7)
        max_tokens (int): Maximum tokens in response (default: 2000)
    
    Returns:
        str: The generated answer
    """
    try:
        client = genai.Client(api_key=gemini_api_key)
        
        response = client.models.generate_content(
            model=model,
            contents=question,
            config=genai.types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
        )
        
        return response.text
    except Exception as e:
        return f"Error calling Gemini API: {str(e)}"


def ask_model(question, model=None, temperature=0.7, max_tokens=2000):
    """
    General function to ask any AI model a question.
    Automatically routes to the appropriate API based on the model name.
    
    Args:
        question (str): The question to ask
        model (str): The model to use. Can be:
            - GPT models: "gpt-4o", "gpt-4", "gpt-3.5-turbo", etc.
            - Claude models: "claude-3-5-sonnet-20241022", "claude-3-opus-20240229", etc.
            - Gemini models: "gemini-2.0-flash-exp", "gemini-1.5-pro", "gemini-1.5-flash", etc.
            - If None, defaults to "gpt-4o"
        temperature (float): Sampling temperature (default: 0.7)
        max_tokens (int): Maximum tokens in response (default: 2000)
    
    Returns:
        str: The generated answer
    """
    # Default model if none specified
    if model is None:
        model = "gpt-4o"
    
    # Determine which API to use based on model name
    model_lower = model.lower()
    
    if "gpt" in model_lower or model_lower.startswith("o1"):
        # OpenAI GPT models
        return ask_gpt(question, model=model, temperature=temperature, max_tokens=max_tokens)
    
    elif "claude" in model_lower:
        # Anthropic Claude models
        return ask_claude(question, model=model, temperature=temperature, max_tokens=max_tokens)
    
    elif "gemini" in model_lower:
        # Google Gemini models
        return ask_gemini(question, model=model, temperature=temperature, max_tokens=max_tokens)
    
    else:
        return f"Error: Unknown model '{model}'. Please specify a GPT, Claude, or Gemini model."
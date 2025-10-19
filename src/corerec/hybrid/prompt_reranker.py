"""
Prompt-based reranker for recommendation systems.

This module provides a prompt-based reranker that uses LLMs to rerank items
based on user preferences and context.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any, Union, Optional, Callable
import logging
import time
import os
import json
import requests
import asyncio
from concurrent.futures import ThreadPoolExecutor
import aiohttp

from corerec.core.base_model import BaseModel


class PromptTemplate:
    """
    Prompt template for LLM-based reranking.
    
    This class provides utilities for creating prompts for LLMs.
    
    Attributes:
        template (str): Prompt template with placeholders
        placeholders (List[str]): List of placeholders in the template
    """
    
    def __init__(self, template: str):
        """Initialize the prompt template.
        
        Args:
            template (str): Prompt template with placeholders in the format "{placeholder}"
        """
        self.template = template
        self.placeholders = [p.strip('{}') for p in template.split('{') if '}' in p]
    
    def format(self, **kwargs) -> str:
        """Format the prompt template with the provided values.
        
        Args:
            **kwargs: Values for the placeholders
            
        Returns:
            str: Formatted prompt
            
        Raises:
            ValueError: If a required placeholder is missing
        """
        # Check if all placeholders are provided
        missing = [p for p in self.placeholders if p not in kwargs]
        if missing:
            raise ValueError(f"Missing values for placeholders: {missing}")
        
        # Format the template
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing value for placeholder: {e}")


class LLMClient:
    """
    Abstract client for LLM APIs.
    
    This class provides an abstract interface for LLM API clients.
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = 'default'):
        """Initialize the LLM client.
        
        Args:
            api_key (Optional[str]): API key for the LLM service
            model_name (str): Name of the model to use
        """
        self.api_key = api_key or os.environ.get('LLM_API_KEY', '')
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
        
        if not self.api_key:
            self.logger.warning("No API key provided. Some LLM services may not work.")
    
    async def generate_async(self, prompt: str, **kwargs) -> str:
        """Generate text from the prompt asynchronously.
        
        Args:
            prompt (str): Prompt for the LLM
            **kwargs: Additional parameters for the LLM
            
        Returns:
            str: Generated text
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from the prompt.
        
        Args:
            prompt (str): Prompt for the LLM
            **kwargs: Additional parameters for the LLM
            
        Returns:
            str: Generated text
        """
        # Use asyncio in a synchronous context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(self.generate_async(prompt, **kwargs))
        finally:
            loop.close()
        return result
    
    async def batch_generate_async(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate text from multiple prompts asynchronously.
        
        Args:
            prompts (List[str]): List of prompts for the LLM
            **kwargs: Additional parameters for the LLM
            
        Returns:
            List[str]: List of generated texts
        """
        # Use asyncio.gather to run multiple generate_async calls concurrently
        tasks = [self.generate_async(prompt, **kwargs) for prompt in prompts]
        return await asyncio.gather(*tasks)
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate text from multiple prompts.
        
        Args:
            prompts (List[str]): List of prompts for the LLM
            **kwargs: Additional parameters for the LLM
            
        Returns:
            List[str]: List of generated texts
        """
        # Use asyncio in a synchronous context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(self.batch_generate_async(prompts, **kwargs))
        finally:
            loop.close()
        return result


class OpenAIClient(LLMClient):
    """
    Client for OpenAI's API.
    
    This class provides a client for OpenAI's API.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = 'gpt-3.5-turbo',
        temperature: float = 0.0,
        max_tokens: int = 100,
        base_url: Optional[str] = None
    ):
        """Initialize the OpenAI client.
        
        Args:
            api_key (Optional[str]): OpenAI API key
            model_name (str): Name of the model to use
            temperature (float): Temperature for sampling
            max_tokens (int): Maximum number of tokens to generate
            base_url (Optional[str]): Base URL for the API
        """
        super().__init__(api_key, model_name)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url or 'https://api.openai.com/v1/chat/completions'
    
    async def generate_async(self, prompt: str, **kwargs) -> str:
        """Generate text from the prompt asynchronously.
        
        Args:
            prompt (str): Prompt for the LLM
            **kwargs: Additional parameters for the LLM
            
        Returns:
            str: Generated text
        """
        # Merge default parameters with provided parameters
        params = {
            'model': self.model_name,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
        }
        params.update(kwargs)
        
        # Create messages
        messages = [
            {'role': 'system', 'content': 'You are a recommendation assistant. Output your choice directly with no explanation or additional content.'},
            {'role': 'user', 'content': prompt}
        ]
        
        # Create request data
        data = {
            'messages': messages,
            **params
        }
        
        # Set up headers
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        # Make request
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(self.base_url, json=data, headers=headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        self.logger.error(f"OpenAI API error: {error_text}")
                        return ""
                    
                    result = await response.json()
                    return result['choices'][0]['message']['content'].strip()
            except Exception as e:
                self.logger.error(f"Error calling OpenAI API: {e}")
                return ""


class AnthropicClient(LLMClient):
    """
    Client for Anthropic's Claude API.
    
    This class provides a client for Anthropic's Claude API.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = 'claude-3-sonnet-20240229',
        temperature: float = 0.0,
        max_tokens: int = 100,
        base_url: Optional[str] = None
    ):
        """Initialize the Anthropic client.
        
        Args:
            api_key (Optional[str]): Anthropic API key
            model_name (str): Name of the model to use
            temperature (float): Temperature for sampling
            max_tokens (int): Maximum number of tokens to generate
            base_url (Optional[str]): Base URL for the API
        """
        super().__init__(api_key, model_name)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url or 'https://api.anthropic.com/v1/messages'
    
    async def generate_async(self, prompt: str, **kwargs) -> str:
        """Generate text from the prompt asynchronously.
        
        Args:
            prompt (str): Prompt for the LLM
            **kwargs: Additional parameters for the LLM
            
        Returns:
            str: Generated text
        """
        # Merge default parameters with provided parameters
        params = {
            'model': self.model_name,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
        }
        params.update(kwargs)
        
        # Create system prompt
        system = "You are a recommendation assistant. Output your choice directly with no explanation or additional content."
        
        # Create request data
        data = {
            'messages': [
                {'role': 'user', 'content': prompt}
            ],
            'system': system,
            **params
        }
        
        # Set up headers
        headers = {
            'x-api-key': self.api_key,
            'anthropic-version': '2023-06-01',
            'Content-Type': 'application/json'
        }
        
        # Make request
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(self.base_url, json=data, headers=headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        self.logger.error(f"Anthropic API error: {error_text}")
                        return ""
                    
                    result = await response.json()
                    return result['content'][0]['text'].strip()
            except Exception as e:
                self.logger.error(f"Error calling Anthropic API: {e}")
                return ""


class PromptReranker(BaseModel):
    """
    Prompt-based reranker for recommendation systems.
    
    This model uses LLMs to rerank items based on user preferences and context.
    
    Attributes:
        name (str): Name of the model
        config (Dict[str, Any]): Model configuration
        llm_client (LLMClient): Client for LLM API
        prompt_template (PromptTemplate): Prompt template for reranking
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialize the prompt-based reranker.
        
        Args:
            name (str): Name of the model
            config (Dict[str, Any]): Model configuration including:
                - llm_type (str): Type of LLM to use ('openai' or 'anthropic')
                - llm_config (Dict[str, Any]): Configuration for the LLM client
                - prompt_template (str): Prompt template for reranking
                - scorer_config (Dict[str, Any]): Configuration for the item scorer
        """
        super().__init__(name, config)
        
        # Create LLM client
        llm_type = config.get('llm_type', 'openai').lower()
        llm_config = config.get('llm_config', {})
        
        if llm_type == 'openai':
            self.llm_client = OpenAIClient(**llm_config)
        elif llm_type == 'anthropic':
            self.llm_client = AnthropicClient(**llm_config)
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")
        
        # Create prompt template
        prompt_template = config.get('prompt_template', "")
        if not prompt_template:
            # Default template
            prompt_template = """
            You are an expert recommendation system.
            The user's preferences are: {user_preferences}
            The current context is: {context}
            Here are the candidate items:
            {items}
            
            Rank these items based on the user's preferences and context. Output ONLY the item ID of the best item.
            """
        
        self.prompt_template = PromptTemplate(prompt_template)
        
        # Additional configuration
        self.scorer_config = config.get('scorer_config', {})
        self.use_async = config.get('use_async', True)
        self.batch_size = config.get('batch_size', 10)
        self.max_candidates = config.get('max_candidates', 100)
    
    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Forward pass of the prompt-based reranker.
        
        Args:
            batch (Dict[str, Any]): Batch of data including:
                - user_preferences (List[str]): List of user preferences
                - context (List[str]): List of context descriptions
                - items (List[List[Dict[str, Any]]]): List of lists of candidate items
                - item_ids (List[List[str]]): List of lists of item IDs
            
        Returns:
            torch.Tensor: Scores for each item
        """
        # Extract batch data
        batch_size = len(batch['user_preferences'])
        
        # Initialize scores
        scores = torch.zeros(batch_size, dtype=torch.float32)
        
        # Process each instance in the batch
        for i in range(batch_size):
            user_prefs = batch['user_preferences'][i]
            context = batch['context'][i]
            items = batch['items'][i]
            item_ids = batch['item_ids'][i]
            
            # Limit number of candidates
            if len(items) > self.max_candidates:
                items = items[:self.max_candidates]
                item_ids = item_ids[:self.max_candidates]
            
            # Format items as string
            items_str = "\n".join([f"{item_id}: {item['description']}" for item_id, item in zip(item_ids, items)])
            
            # Format prompt
            prompt = self.prompt_template.format(
                user_preferences=user_prefs,
                context=context,
                items=items_str
            )
            
            # Generate response
            response = self.llm_client.generate(prompt)
            
            # Extract item ID from response
            try:
                # Find the item ID in the response
                for item_id in item_ids:
                    if item_id in response:
                        idx = item_ids.index(item_id)
                        scores[i] = 1.0
                        break
            except Exception as e:
                self.logger.error(f"Error parsing response: {e}")
        
        return scores
    
    async def rerank_async(
        self,
        user_preferences: str,
        context: str,
        items: List[Dict[str, Any]],
        item_ids: List[str]
    ) -> List[Tuple[str, float]]:
        """Rerank items asynchronously.
        
        Args:
            user_preferences (str): User preferences
            context (str): Context description
            items (List[Dict[str, Any]]): List of candidate items
            item_ids (List[str]): List of item IDs
            
        Returns:
            List[Tuple[str, float]]: List of item IDs and scores, sorted by score
        """
        # Limit number of candidates
        if len(items) > self.max_candidates:
            items = items[:self.max_candidates]
            item_ids = item_ids[:self.max_candidates]
        
        # Format items as string
        items_str = "\n".join([f"{item_id}: {item['description']}" for item_id, item in zip(item_ids, items)])
        
        # Format prompt
        prompt = self.prompt_template.format(
            user_preferences=user_preferences,
            context=context,
            items=items_str
        )
        
        # Generate response
        response = await self.llm_client.generate_async(prompt)
        
        # Extract item ID from response and create scores
        reranked = []
        try:
            # Find the item ID in the response
            for item_id in item_ids:
                if item_id in response:
                    reranked.append((item_id, 1.0))
                else:
                    reranked.append((item_id, 0.0))
        except Exception as e:
            self.logger.error(f"Error parsing response: {e}")
            # Fall back to original order
            reranked = [(item_id, 0.0) for item_id in item_ids]
        
        # Sort by score
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        return reranked
    
    def rerank(
        self,
        user_preferences: str,
        context: str,
        items: List[Dict[str, Any]],
        item_ids: List[str]
    ) -> List[Tuple[str, float]]:
        """Rerank items.
        
        Args:
            user_preferences (str): User preferences
            context (str): Context description
            items (List[Dict[str, Any]]): List of candidate items
            item_ids (List[str]): List of item IDs
            
        Returns:
            List[Tuple[str, float]]: List of item IDs and scores, sorted by score
        """
        # Use asyncio in a synchronous context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(self.rerank_async(user_preferences, context, items, item_ids))
        finally:
            loop.close()
        return result
    
    async def batch_rerank_async(
        self,
        user_preferences_list: List[str],
        context_list: List[str],
        items_list: List[List[Dict[str, Any]]],
        item_ids_list: List[List[str]]
    ) -> List[List[Tuple[str, float]]]:
        """Rerank multiple sets of items asynchronously.
        
        Args:
            user_preferences_list (List[str]): List of user preferences
            context_list (List[str]): List of context descriptions
            items_list (List[List[Dict[str, Any]]]): List of lists of candidate items
            item_ids_list (List[List[str]]): List of lists of item IDs
            
        Returns:
            List[List[Tuple[str, float]]]: List of lists of item IDs and scores, sorted by score
        """
        # Create tasks for each set of items
        tasks = []
        for i in range(len(user_preferences_list)):
            task = self.rerank_async(
                user_preferences_list[i],
                context_list[i],
                items_list[i],
                item_ids_list[i]
            )
            tasks.append(task)
        
        # Run tasks concurrently
        return await asyncio.gather(*tasks)
    
    def batch_rerank(
        self,
        user_preferences_list: List[str],
        context_list: List[str],
        items_list: List[List[Dict[str, Any]]],
        item_ids_list: List[List[str]]
    ) -> List[List[Tuple[str, float]]]:
        """Rerank multiple sets of items.
        
        Args:
            user_preferences_list (List[str]): List of user preferences
            context_list (List[str]): List of context descriptions
            items_list (List[List[Dict[str, Any]]]): List of lists of candidate items
            item_ids_list (List[List[str]]): List of lists of item IDs
            
        Returns:
            List[List[Tuple[str, float]]]: List of lists of item IDs and scores, sorted by score
        """
        # Use asyncio in a synchronous context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                self.batch_rerank_async(user_preferences_list, context_list, items_list, item_ids_list)
            )
        finally:
            loop.close()
        return result
    
    def train_step(self, batch: Dict[str, Any], optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """Perform a single training step.
        
        This model does not support training.
        
        Args:
            batch (Dict[str, Any]): Batch of data
            optimizer (torch.optim.Optimizer): Optimizer instance
            
        Returns:
            Dict[str, float]: Dictionary with loss values
        """
        self.logger.warning("PromptReranker does not support training")
        return {'loss': 0.0}
    
    def validate_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Perform a single validation step.
        
        This model does not support validation.
        
        Args:
            batch (Dict[str, Any]): Batch of data
            
        Returns:
            Dict[str, float]: Dictionary with validation metrics
        """
        self.logger.warning("PromptReranker does not support validation")
        return {'val_loss': 0.0} 
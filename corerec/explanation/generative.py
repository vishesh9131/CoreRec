"""
Generative Explanations using LLMs

Generate natural language explanations using language models.
More flexible and natural than template-based approaches.
"""

from typing import Any, Callable, Dict, List, Optional

from .base import BaseExplainer, Explanation


class GenerativeExplainer(BaseExplainer):
    """
    Generates explanations using a language model.
    
    Can use any LLM (OpenAI, local models, etc) to generate
    natural, context-aware explanations.
    
    Example:
        explainer = GenerativeExplainer(
            llm=openai_client,
            item_context_fn=get_item_info,
            user_context_fn=get_user_profile,
        )
        
        explanation = explainer.explain(item_id=123, context={'user_id': 456})
        # "Based on your interest in science fiction and recent views of 
        #  Blade Runner, you might enjoy this cyberpunk thriller..."
    """
    
    def __init__(
        self,
        llm: Optional[Any] = None,
        llm_fn: Optional[Callable[[str], str]] = None,
        item_context_fn: Optional[Callable[[Any], Dict[str, Any]]] = None,
        user_context_fn: Optional[Callable[[Any], Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        max_length: int = 100,
        name: str = "generative_explainer",
    ):
        """
        Args:
            llm: LLM client with a generate/complete method
            llm_fn: simple function(prompt) -> response (alternative to client)
            item_context_fn: function(item_id) -> item context dict
            user_context_fn: function(user_id) -> user context dict
            system_prompt: system prompt for the LLM
            max_length: max explanation length
            name: identifier
        """
        super().__init__(name=name)
        
        self.llm = llm
        self.llm_fn = llm_fn
        self.item_context_fn = item_context_fn
        self.user_context_fn = user_context_fn
        self.max_length = max_length
        
        self.system_prompt = system_prompt or (
            "You are a helpful recommendation assistant. "
            "Generate a brief, natural explanation for why an item "
            "was recommended to a user. Be concise and specific. "
            "Do not use generic phrases like 'you might like this'. "
            "Focus on the connection between the user's preferences and the item."
        )
    
    def explain(
        self,
        item_id: Any,
        context: Dict[str, Any],
        **kwargs
    ) -> Explanation:
        """Generate explanation using LLM."""
        if self.llm is None and self.llm_fn is None:
            # no LLM available, return generic explanation
            return Explanation(
                item_id=item_id,
                text="Recommended for you",
                explanation_type="generic",
            )
        
        user_id = context.get('user_id')
        
        # gather context
        item_info = {}
        user_info = {}
        
        if self.item_context_fn:
            item_info = self.item_context_fn(item_id) or {}
        
        if self.user_context_fn and user_id:
            user_info = self.user_context_fn(user_id) or {}
        
        # build prompt
        prompt = self._build_prompt(item_id, item_info, user_info, context)
        
        # generate
        try:
            response = self._call_llm(prompt)
            text = self._clean_response(response)
        except Exception as e:
            text = "Recommended based on your preferences"
        
        return Explanation(
            item_id=item_id,
            text=text,
            explanation_type="generative",
            features={'item_info': item_info, 'user_info': user_info},
        )
    
    def _build_prompt(
        self,
        item_id: Any,
        item_info: Dict,
        user_info: Dict,
        context: Dict
    ) -> str:
        """Build the prompt for the LLM."""
        parts = []
        
        # item description
        if item_info:
            item_desc = ", ".join(f"{k}: {v}" for k, v in item_info.items())
            parts.append(f"Item: {item_desc}")
        else:
            parts.append(f"Item ID: {item_id}")
        
        # user description
        if user_info:
            user_desc = ", ".join(f"{k}: {v}" for k, v in user_info.items())
            parts.append(f"User preferences: {user_desc}")
        
        # recommendation context
        if context.get('source'):
            parts.append(f"Recommendation source: {context['source']}")
        
        parts.append(
            f"Generate a brief explanation (max {self.max_length} characters) "
            "for why this item was recommended."
        )
        
        return "\n".join(parts)
    
    def _call_llm(self, prompt: str) -> str:
        """Call the LLM to generate response."""
        if self.llm_fn:
            return self.llm_fn(prompt)
        
        # try common LLM client interfaces
        if hasattr(self.llm, 'generate'):
            return self.llm.generate(prompt)
        elif hasattr(self.llm, 'complete'):
            return self.llm.complete(prompt)
        elif hasattr(self.llm, 'chat'):
            # OpenAI-style
            response = self.llm.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
            )
            return response.choices[0].message.content
        elif callable(self.llm):
            return self.llm(prompt)
        
        raise ValueError("Could not determine how to call the LLM")
    
    def _clean_response(self, response: str) -> str:
        """Clean up LLM response."""
        # remove quotes
        response = response.strip().strip('"\'')
        
        # truncate if too long
        if len(response) > self.max_length:
            response = response[:self.max_length - 3] + "..."
        
        return response
    
    def explain_batch(
        self,
        item_ids: List[Any],
        context: Dict[str, Any],
        **kwargs
    ) -> List[Explanation]:
        """
        Batch explanation generation.
        
        For LLMs that support batching, this can be more efficient.
        """
        # for now, just loop
        # future: batch prompts together
        return [self.explain(item_id, context, **kwargs) for item_id in item_ids]

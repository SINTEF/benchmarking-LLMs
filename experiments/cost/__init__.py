"""Up-front token & cost estimation for LLM entity-linking runs.

The estimator rebuilds the exact prompts LionLinker would send (same template,
table context window and candidate formatting), counts their tokens and prices
them with live OpenRouter rates. Nothing here calls an LLM.
"""

"""Tool: ml_transformers
Hugging Face Transformers for NLP, text generation, and embeddings.

Supported operations:
- text_generation: Generate text with language models
- text_classification: Classify text into categories
- sentiment_analysis: Analyze sentiment of text
- ner: Named entity recognition
- question_answering: Answer questions from context
- summarization: Summarize long text
- translation: Translate between languages
- embeddings: Generate text embeddings
- zero_shot: Zero-shot classification
"""
from typing import Any, Dict, List, Optional
import json


def _optional_import(module_name: str):
    try:
        return __import__(module_name)
    except ImportError:
        return None


transformers = _optional_import("transformers")
torch = _optional_import("torch")

# Pipeline cache
_pipeline_cache: Dict[str, Any] = {}


def _get_device() -> str:
    """Get available device."""
    if torch and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _get_pipeline(task: str, model: Optional[str] = None) -> Any:
    """Get or create a pipeline."""
    if transformers is None:
        raise ImportError("transformers is not installed. Run: pip install transformers")
    
    from transformers import pipeline
    
    cache_key = f"{task}:{model or 'default'}"
    
    if cache_key not in _pipeline_cache:
        kwargs = {"task": task, "device": _get_device()}
        if model:
            kwargs["model"] = model
        _pipeline_cache[cache_key] = pipeline(**kwargs)
    
    return _pipeline_cache[cache_key]


def _text_generation(
    prompt: str,
    model: Optional[str] = None,
    max_length: int = 100,
    num_return_sequences: int = 1,
    temperature: float = 1.0,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> Dict[str, Any]:
    """Generate text from a prompt."""
    pipe = _get_pipeline("text-generation", model)
    
    results = pipe(
        prompt,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
    )
    
    return {
        "generated": [r["generated_text"] for r in results],
        "model": model or "default",
    }


def _text_classification(
    texts: List[str],
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Classify text into categories."""
    pipe = _get_pipeline("text-classification", model)
    results = pipe(texts)
    
    return {
        "classifications": results if isinstance(results[0], dict) else [results],
    }


def _sentiment_analysis(
    texts: List[str],
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Analyze sentiment of text."""
    model = model or "distilbert-base-uncased-finetuned-sst-2-english"
    pipe = _get_pipeline("sentiment-analysis", model)
    results = pipe(texts)
    
    return {
        "sentiments": results if isinstance(results, list) else [results],
    }


def _named_entity_recognition(
    text: str,
    model: Optional[str] = None,
    aggregation_strategy: str = "simple",
) -> Dict[str, Any]:
    """Extract named entities from text."""
    pipe = _get_pipeline("ner", model)
    results = pipe(text, aggregation_strategy=aggregation_strategy)
    
    return {
        "entities": [
            {
                "entity": r.get("entity_group", r.get("entity", "")),
                "word": r.get("word", ""),
                "score": float(r.get("score", 0)),
                "start": r.get("start"),
                "end": r.get("end"),
            }
            for r in results
        ],
    }


def _question_answering(
    question: str,
    context: str,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Answer question from context."""
    pipe = _get_pipeline("question-answering", model)
    result = pipe(question=question, context=context)
    
    return {
        "answer": result["answer"],
        "score": float(result["score"]),
        "start": result["start"],
        "end": result["end"],
    }


def _summarization(
    text: str,
    model: Optional[str] = None,
    max_length: int = 130,
    min_length: int = 30,
) -> Dict[str, Any]:
    """Summarize long text."""
    pipe = _get_pipeline("summarization", model)
    result = pipe(text, max_length=max_length, min_length=min_length)
    
    return {
        "summary": result[0]["summary_text"],
    }


def _translation(
    text: str,
    model: Optional[str] = None,
    src_lang: str = "en",
    tgt_lang: str = "fr",
) -> Dict[str, Any]:
    """Translate text between languages."""
    # Use appropriate model based on language pair
    if model is None:
        model = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    
    pipe = _get_pipeline("translation", model)
    result = pipe(text)
    
    return {
        "translation": result[0]["translation_text"],
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
    }


def _embeddings(
    texts: List[str],
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate text embeddings."""
    if transformers is None:
        raise ImportError("transformers is not installed")
    
    from transformers import AutoTokenizer, AutoModel
    
    model_name = model or "sentence-transformers/all-MiniLM-L6-v2"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_obj = AutoModel.from_pretrained(model_name)
    
    if torch and torch.cuda.is_available():
        model_obj = model_obj.to("cuda")
    
    encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    
    if torch and torch.cuda.is_available():
        encoded = {k: v.to("cuda") for k, v in encoded.items()}
    
    with torch.no_grad() if torch else nullcontext():
        outputs = model_obj(**encoded)
        # Mean pooling
        embeddings = outputs.last_hidden_state.mean(dim=1)
    
    return {
        "embeddings": embeddings.cpu().tolist() if torch else embeddings.tolist(),
        "dimension": embeddings.shape[-1],
    }


def _zero_shot_classification(
    text: str,
    candidate_labels: List[str],
    model: Optional[str] = None,
    multi_label: bool = False,
) -> Dict[str, Any]:
    """Zero-shot text classification."""
    model = model or "facebook/bart-large-mnli"
    pipe = _get_pipeline("zero-shot-classification", model)
    
    result = pipe(text, candidate_labels, multi_label=multi_label)
    
    return {
        "sequence": result["sequence"],
        "labels": result["labels"],
        "scores": [float(s) for s in result["scores"]],
    }


class nullcontext:
    """Null context manager for when torch is not available."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


def run(args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run Transformers operations.
    
    Args:
        args: Dictionary with:
            - operation: NLP task to perform
            - model: Optional model name/path
            - text/texts: Input text(s)
            - Various operation-specific parameters
    
    Returns:
        Result dictionary with operation output
    """
    args = args or {}
    operation = args.get("operation", "sentiment_analysis")
    model = args.get("model")
    
    if transformers is None:
        return {
            "tool": "ml_transformers",
            "status": "error",
            "error": "transformers not installed. Run: pip install transformers torch",
        }
    
    try:
        if operation == "text_generation":
            result = _text_generation(
                prompt=args.get("prompt", ""),
                model=model,
                max_length=args.get("max_length", 100),
                num_return_sequences=args.get("num_return_sequences", 1),
                temperature=args.get("temperature", 1.0),
                top_p=args.get("top_p", 0.9),
                do_sample=args.get("do_sample", True),
            )
        
        elif operation == "text_classification":
            result = _text_classification(
                texts=args.get("texts", []),
                model=model,
            )
        
        elif operation == "sentiment_analysis":
            result = _sentiment_analysis(
                texts=args.get("texts", []),
                model=model,
            )
        
        elif operation == "ner":
            result = _named_entity_recognition(
                text=args.get("text", ""),
                model=model,
                aggregation_strategy=args.get("aggregation_strategy", "simple"),
            )
        
        elif operation == "question_answering":
            result = _question_answering(
                question=args.get("question", ""),
                context=args.get("context", ""),
                model=model,
            )
        
        elif operation == "summarization":
            result = _summarization(
                text=args.get("text", ""),
                model=model,
                max_length=args.get("max_length", 130),
                min_length=args.get("min_length", 30),
            )
        
        elif operation == "translation":
            result = _translation(
                text=args.get("text", ""),
                model=model,
                src_lang=args.get("src_lang", "en"),
                tgt_lang=args.get("tgt_lang", "fr"),
            )
        
        elif operation == "embeddings":
            result = _embeddings(
                texts=args.get("texts", []),
                model=model,
            )
        
        elif operation == "zero_shot":
            result = _zero_shot_classification(
                text=args.get("text", ""),
                candidate_labels=args.get("candidate_labels", []),
                model=model,
                multi_label=args.get("multi_label", False),
            )
        
        else:
            return {
                "tool": "ml_transformers",
                "status": "error",
                "error": f"Unknown operation: {operation}",
            }
        
        return {"tool": "ml_transformers", "status": "ok", **result}
    
    except Exception as e:
        return {"tool": "ml_transformers", "status": "error", "error": str(e)}


def example():
    """Example usage payloads."""
    return {
        "sentiment": {
            "operation": "sentiment_analysis",
            "texts": ["I love this product!", "This is terrible."],
        },
        "ner": {
            "operation": "ner",
            "text": "Apple Inc. was founded by Steve Jobs in Cupertino, California.",
        },
        "qa": {
            "operation": "question_answering",
            "question": "What is the capital of France?",
            "context": "Paris is the capital and largest city of France.",
        },
        "summarize": {
            "operation": "summarization",
            "text": "Long article text here...",
            "max_length": 100,
        },
        "zero_shot": {
            "operation": "zero_shot",
            "text": "I just bought a new smartphone and it's amazing!",
            "candidate_labels": ["technology", "sports", "politics", "entertainment"],
        },
    }

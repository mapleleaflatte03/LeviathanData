"""Tool: mm_openclip
OpenCLIP for image and text embeddings.

Supported operations:
- embed_image: Generate image embeddings
- embed_text: Generate text embeddings
- similarity: Compute image-text similarity
- zero_shot_classify: Zero-shot image classification
- list_models: List available models
"""
from typing import Any, Dict, List, Optional, Union
import json
import os


def _optional_import(module_name: str):
    try:
        return __import__(module_name)
    except ImportError:
        return None


open_clip = _optional_import("open_clip")
torch = _optional_import("torch")
PIL = _optional_import("PIL")

# Model cache
_model_cache: Dict[str, Any] = {}


def _get_device() -> str:
    """Get available device."""
    if torch and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _get_model(
    model_name: str = "ViT-B-32",
    pretrained: str = "laion2b_s34b_b79k",
) -> tuple:
    """Get or create model and preprocessing."""
    if open_clip is None:
        raise ImportError("open_clip is not installed. Run: pip install open_clip_torch")
    
    cache_key = f"{model_name}:{pretrained}"
    
    if cache_key not in _model_cache:
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        tokenizer = open_clip.get_tokenizer(model_name)
        
        device = _get_device()
        model = model.to(device)
        model.eval()
        
        _model_cache[cache_key] = {
            "model": model,
            "preprocess": preprocess,
            "tokenizer": tokenizer,
            "device": device,
        }
    
    return _model_cache[cache_key]


def _load_image(image_path: str) -> Any:
    """Load image from path."""
    if PIL is None:
        raise ImportError("Pillow is not installed. Run: pip install Pillow")
    
    from PIL import Image
    return Image.open(image_path).convert("RGB")


def _embed_image(
    image_paths: List[str],
    model_name: str = "ViT-B-32",
    pretrained: str = "laion2b_s34b_b79k",
    normalize: bool = True,
) -> Dict[str, Any]:
    """Generate image embeddings."""
    model_info = _get_model(model_name, pretrained)
    model = model_info["model"]
    preprocess = model_info["preprocess"]
    device = model_info["device"]
    
    images = [preprocess(_load_image(p)).unsqueeze(0) for p in image_paths]
    images = torch.cat(images).to(device)
    
    with torch.no_grad():
        embeddings = model.encode_image(images)
        if normalize:
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
    
    return {
        "embeddings": embeddings.cpu().numpy().tolist(),
        "dimension": embeddings.shape[-1],
        "count": len(image_paths),
    }


def _embed_text(
    texts: List[str],
    model_name: str = "ViT-B-32",
    pretrained: str = "laion2b_s34b_b79k",
    normalize: bool = True,
) -> Dict[str, Any]:
    """Generate text embeddings."""
    model_info = _get_model(model_name, pretrained)
    model = model_info["model"]
    tokenizer = model_info["tokenizer"]
    device = model_info["device"]
    
    tokens = tokenizer(texts).to(device)
    
    with torch.no_grad():
        embeddings = model.encode_text(tokens)
        if normalize:
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
    
    return {
        "embeddings": embeddings.cpu().numpy().tolist(),
        "dimension": embeddings.shape[-1],
        "count": len(texts),
    }


def _similarity(
    image_paths: List[str],
    texts: List[str],
    model_name: str = "ViT-B-32",
    pretrained: str = "laion2b_s34b_b79k",
) -> Dict[str, Any]:
    """Compute image-text similarity."""
    model_info = _get_model(model_name, pretrained)
    model = model_info["model"]
    preprocess = model_info["preprocess"]
    tokenizer = model_info["tokenizer"]
    device = model_info["device"]
    
    images = [preprocess(_load_image(p)).unsqueeze(0) for p in image_paths]
    images = torch.cat(images).to(device)
    tokens = tokenizer(texts).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(images)
        text_features = model.encode_text(tokens)
        
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        similarity = (image_features @ text_features.T).cpu().numpy()
    
    return {
        "similarity_matrix": similarity.tolist(),
        "image_count": len(image_paths),
        "text_count": len(texts),
    }


def _zero_shot_classify(
    image_path: str,
    labels: List[str],
    model_name: str = "ViT-B-32",
    pretrained: str = "laion2b_s34b_b79k",
    prompt_template: str = "a photo of a {}",
) -> Dict[str, Any]:
    """Zero-shot image classification."""
    model_info = _get_model(model_name, pretrained)
    model = model_info["model"]
    preprocess = model_info["preprocess"]
    tokenizer = model_info["tokenizer"]
    device = model_info["device"]
    
    image = preprocess(_load_image(image_path)).unsqueeze(0).to(device)
    
    text_prompts = [prompt_template.format(label) for label in labels]
    tokens = tokenizer(text_prompts).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(tokens)
        
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        logits = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        probs = logits.cpu().numpy()[0]
    
    results = sorted(zip(labels, probs.tolist()), key=lambda x: x[1], reverse=True)
    
    return {
        "predictions": [{"label": label, "probability": prob} for label, prob in results],
        "top_label": results[0][0],
        "top_probability": results[0][1],
    }


def _list_models() -> Dict[str, Any]:
    """List available models."""
    if open_clip is None:
        raise ImportError("open_clip is not installed")
    
    models = open_clip.list_pretrained()
    
    # Group by model name
    model_dict = {}
    for model_name, pretrained in models:
        if model_name not in model_dict:
            model_dict[model_name] = []
        model_dict[model_name].append(pretrained)
    
    return {
        "models": model_dict,
        "total_variants": len(models),
    }


def run(args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run OpenCLIP operations."""
    args = args or {}
    operation = args.get("operation", "list_models")
    
    if open_clip is None:
        return {
            "tool": "mm_openclip",
            "status": "error",
            "error": "open_clip not installed. Run: pip install open_clip_torch",
        }
    
    try:
        model_kwargs = {
            "model_name": args.get("model_name", "ViT-B-32"),
            "pretrained": args.get("pretrained", "laion2b_s34b_b79k"),
        }
        
        if operation == "embed_image":
            result = _embed_image(
                image_paths=args.get("image_paths", []),
                normalize=args.get("normalize", True),
                **model_kwargs,
            )
        
        elif operation == "embed_text":
            result = _embed_text(
                texts=args.get("texts", []),
                normalize=args.get("normalize", True),
                **model_kwargs,
            )
        
        elif operation == "similarity":
            result = _similarity(
                image_paths=args.get("image_paths", []),
                texts=args.get("texts", []),
                **model_kwargs,
            )
        
        elif operation == "zero_shot":
            result = _zero_shot_classify(
                image_path=args.get("image_path", ""),
                labels=args.get("labels", []),
                prompt_template=args.get("prompt_template", "a photo of a {}"),
                **model_kwargs,
            )
        
        elif operation == "list_models":
            result = _list_models()
        
        else:
            return {"tool": "mm_openclip", "status": "error", "error": f"Unknown operation: {operation}"}
        
        return {"tool": "mm_openclip", "status": "ok", **result}
    
    except Exception as e:
        return {"tool": "mm_openclip", "status": "error", "error": str(e)}


def example():
    """Example usage payloads."""
    return {
        "embed_images": {
            "operation": "embed_image",
            "image_paths": ["image1.jpg", "image2.jpg"],
            "model_name": "ViT-B-32",
        },
        "embed_texts": {
            "operation": "embed_text",
            "texts": ["a dog", "a cat", "a bird"],
        },
        "zero_shot": {
            "operation": "zero_shot",
            "image_path": "test.jpg",
            "labels": ["dog", "cat", "bird", "fish"],
        },
        "similarity": {
            "operation": "similarity",
            "image_paths": ["image.jpg"],
            "texts": ["a dog playing", "a cat sleeping", "a bird flying"],
        },
    }

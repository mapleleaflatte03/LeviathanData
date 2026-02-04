"""Tool: ml_pytorch
PyTorch deep learning toolkit.

Supported operations:
- create_model: Create neural network architecture
- train: Train model on data
- predict: Make predictions
- save_model: Save model to file
- load_model: Load model from file
- tensor_ops: Basic tensor operations
"""
from typing import Any, Dict, List, Optional, Tuple
import json
import base64
import io


def _optional_import(module_name: str):
    try:
        return __import__(module_name)
    except ImportError:
        return None


torch = _optional_import("torch")
nn = None
optim = None
F = None

if torch:
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F


class DynamicMLP(nn.Module):
    """Dynamic Multi-Layer Perceptron."""
    
    def __init__(self, layer_sizes: List[int], activation: str = "relu", dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
    
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            if self.activation == "relu":
                x = F.relu(x)
            elif self.activation == "tanh":
                x = torch.tanh(x)
            elif self.activation == "sigmoid":
                x = torch.sigmoid(x)
            elif self.activation == "leaky_relu":
                x = F.leaky_relu(x)
            if self.dropout:
                x = self.dropout(x)
        x = self.layers[-1](x)
        return x


class DynamicCNN(nn.Module):
    """Dynamic Convolutional Neural Network."""
    
    def __init__(
        self,
        input_channels: int,
        conv_layers: List[Dict],
        fc_layers: List[int],
        output_size: int,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        in_channels = input_channels
        for conv in conv_layers:
            out_channels = conv.get("out_channels", 32)
            kernel_size = conv.get("kernel_size", 3)
            padding = conv.get("padding", 1)
            self.convs.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding))
            if conv.get("pool"):
                self.pools.append(nn.MaxPool2d(conv["pool"]))
            else:
                self.pools.append(None)
            in_channels = out_channels
        
        self.flatten = nn.Flatten()
        self.fcs = nn.ModuleList()
        
        for i, fc_size in enumerate(fc_layers):
            if i == 0:
                # Will be set dynamically
                self.fcs.append(nn.LazyLinear(fc_size))
            else:
                self.fcs.append(nn.Linear(fc_layers[i-1], fc_size))
        
        if fc_layers:
            self.fcs.append(nn.Linear(fc_layers[-1], output_size))
        else:
            self.fcs.append(nn.LazyLinear(output_size))
    
    def forward(self, x):
        for conv, pool in zip(self.convs, self.pools):
            x = F.relu(conv(x))
            if pool:
                x = pool(x)
        x = self.flatten(x)
        for fc in self.fcs[:-1]:
            x = F.relu(fc(x))
        x = self.fcs[-1](x)
        return x


class DynamicRNN(nn.Module):
    """Dynamic Recurrent Neural Network (LSTM/GRU)."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        output_size: int = 1,
        rnn_type: str = "lstm",
        bidirectional: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        if rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size, hidden_size, num_layers,
                batch_first=True, bidirectional=bidirectional, dropout=dropout
            )
        elif rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size, hidden_size, num_layers,
                batch_first=True, bidirectional=bidirectional, dropout=dropout
            )
        
        fc_input = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(fc_input, output_size)
    
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out


def _create_model(config: Dict[str, Any]) -> Tuple[nn.Module, str]:
    """Create a model from configuration."""
    model_type = config.get("type", "mlp")
    
    if model_type == "mlp":
        model = DynamicMLP(
            layer_sizes=config.get("layer_sizes", [10, 64, 32, 1]),
            activation=config.get("activation", "relu"),
            dropout=config.get("dropout", 0.0),
        )
    elif model_type == "cnn":
        model = DynamicCNN(
            input_channels=config.get("input_channels", 1),
            conv_layers=config.get("conv_layers", [{"out_channels": 32, "kernel_size": 3, "pool": 2}]),
            fc_layers=config.get("fc_layers", [128]),
            output_size=config.get("output_size", 10),
        )
    elif model_type == "rnn":
        model = DynamicRNN(
            input_size=config.get("input_size", 10),
            hidden_size=config.get("hidden_size", 64),
            num_layers=config.get("num_layers", 1),
            output_size=config.get("output_size", 1),
            rnn_type=config.get("rnn_type", "lstm"),
            bidirectional=config.get("bidirectional", False),
            dropout=config.get("dropout", 0.0),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model, model_type


def _train(
    model_config: Dict,
    X: List[List[float]],
    y: List[Any],
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    optimizer_name: str = "adam",
    loss_fn: str = "mse",
    device: str = "auto",
) -> Dict[str, Any]:
    """Train a model."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model, model_type = _create_model(model_config)
    model = model.to(device)
    
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    
    if loss_fn in ["cross_entropy", "nll"]:
        y_tensor = torch.tensor(y, dtype=torch.long).to(device)
    else:
        y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
        if y_tensor.dim() == 1:
            y_tensor = y_tensor.unsqueeze(1)
    
    # Optimizer
    if optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_name == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Loss function
    if loss_fn == "mse":
        criterion = nn.MSELoss()
    elif loss_fn == "mae":
        criterion = nn.L1Loss()
    elif loss_fn == "cross_entropy":
        criterion = nn.CrossEntropyLoss()
    elif loss_fn == "bce":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.MSELoss()
    
    # Training loop
    history = {"loss": []}
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        history["loss"].append(avg_loss)
    
    # Serialize model
    buffer = io.BytesIO()
    torch.save({
        "model_state": model.state_dict(),
        "model_config": model_config,
    }, buffer)
    model_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    return {
        "model_type": model_type,
        "epochs": epochs,
        "final_loss": history["loss"][-1],
        "history": history,
        "model_b64": model_b64,
        "device": device,
    }


def _predict(model_b64: str, X: List[List[float]], device: str = "auto") -> Dict[str, Any]:
    """Make predictions."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    buffer = io.BytesIO(base64.b64decode(model_b64))
    checkpoint = torch.load(buffer, map_location=device)
    
    model, _ = _create_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    model.eval()
    
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        outputs = model(X_tensor)
        predictions = outputs.cpu().numpy().tolist()
    
    return {"predictions": predictions}


def _tensor_ops(
    operation: str,
    data: Optional[List] = None,
    shape: Optional[List[int]] = None,
    **kwargs
) -> Dict[str, Any]:
    """Basic tensor operations."""
    
    if operation == "create":
        if data:
            tensor = torch.tensor(data, dtype=torch.float32)
        elif shape:
            fill = kwargs.get("fill", "zeros")
            if fill == "zeros":
                tensor = torch.zeros(shape)
            elif fill == "ones":
                tensor = torch.ones(shape)
            elif fill == "rand":
                tensor = torch.rand(shape)
            elif fill == "randn":
                tensor = torch.randn(shape)
            else:
                tensor = torch.zeros(shape)
        else:
            tensor = torch.tensor([0.0])
        
        return {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "data": tensor.tolist(),
        }
    
    elif operation == "matmul":
        a = torch.tensor(kwargs.get("a", []), dtype=torch.float32)
        b = torch.tensor(kwargs.get("b", []), dtype=torch.float32)
        result = torch.matmul(a, b)
        return {"result": result.tolist(), "shape": list(result.shape)}
    
    elif operation == "einsum":
        equation = kwargs.get("equation", "")
        operands = [torch.tensor(op, dtype=torch.float32) for op in kwargs.get("operands", [])]
        result = torch.einsum(equation, *operands)
        return {"result": result.tolist(), "shape": list(result.shape)}
    
    elif operation == "softmax":
        tensor = torch.tensor(data, dtype=torch.float32)
        dim = kwargs.get("dim", -1)
        result = F.softmax(tensor, dim=dim)
        return {"result": result.tolist()}
    
    elif operation == "normalize":
        tensor = torch.tensor(data, dtype=torch.float32)
        dim = kwargs.get("dim", -1)
        result = F.normalize(tensor, dim=dim)
        return {"result": result.tolist()}
    
    else:
        raise ValueError(f"Unknown tensor operation: {operation}")


def run(args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run PyTorch operations.
    
    Args:
        args: Dictionary with:
            - operation: "train", "predict", "tensor_ops", "info"
            - model_config: Model architecture configuration
            - X: Input data
            - y: Target data
            - epochs, batch_size, learning_rate: Training params
    
    Returns:
        Result dictionary with operation output
    """
    args = args or {}
    operation = args.get("operation", "info")
    
    if torch is None:
        return {"tool": "ml_pytorch", "status": "error", "error": "pytorch not installed"}
    
    try:
        if operation == "info":
            return {
                "tool": "ml_pytorch",
                "status": "ok",
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }
        
        elif operation == "train":
            result = _train(
                model_config=args.get("model_config", {"type": "mlp", "layer_sizes": [10, 64, 1]}),
                X=args.get("X", []),
                y=args.get("y", []),
                epochs=args.get("epochs", 100),
                batch_size=args.get("batch_size", 32),
                learning_rate=args.get("learning_rate", 0.001),
                optimizer_name=args.get("optimizer", "adam"),
                loss_fn=args.get("loss_fn", "mse"),
                device=args.get("device", "auto"),
            )
            return {"tool": "ml_pytorch", "status": "ok", **result}
        
        elif operation == "predict":
            result = _predict(
                model_b64=args.get("model_b64", ""),
                X=args.get("X", []),
                device=args.get("device", "auto"),
            )
            return {"tool": "ml_pytorch", "status": "ok", **result}
        
        elif operation == "tensor_ops":
            result = _tensor_ops(
                operation=args.get("tensor_op", "create"),
                data=args.get("data"),
                shape=args.get("shape"),
                **args
            )
            return {"tool": "ml_pytorch", "status": "ok", **result}
        
        else:
            return {"tool": "ml_pytorch", "status": "error", "error": f"Unknown operation: {operation}"}
    
    except Exception as e:
        return {"tool": "ml_pytorch", "status": "error", "error": str(e)}


def example():
    """Example usage payloads."""
    return {
        "train_mlp": {
            "operation": "train",
            "model_config": {
                "type": "mlp",
                "layer_sizes": [2, 64, 32, 1],
                "activation": "relu",
                "dropout": 0.1,
            },
            "X": [[0, 0], [0, 1], [1, 0], [1, 1]],
            "y": [0, 1, 1, 0],
            "epochs": 1000,
            "learning_rate": 0.01,
            "loss_fn": "bce",
        },
        "train_rnn": {
            "operation": "train",
            "model_config": {
                "type": "rnn",
                "input_size": 5,
                "hidden_size": 32,
                "num_layers": 2,
                "output_size": 1,
                "rnn_type": "lstm",
            },
            "X": [[[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7]]],
            "y": [[8]],
            "epochs": 100,
        },
        "tensor_softmax": {
            "operation": "tensor_ops",
            "tensor_op": "softmax",
            "data": [[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]],
            "dim": -1,
        },
    }

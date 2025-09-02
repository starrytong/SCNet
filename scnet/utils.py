from collections import defaultdict
from contextlib import contextmanager
import os
import tempfile
import typing as tp
import numpy as np
import torch
import julius
from pathlib import Path

# Audio
def convert_audio_channels(wav, channels=2):
    """Convert audio to the given number of channels."""
    # Debug info
    print(f"DEBUG: Input tensor shape: {wav.shape}, ndim: {wav.ndim}, requested channels: {channels}")
    
    # Handle 1D arrays (samples only, no channel dimension)
    if wav.ndim == 1:
        print("DEBUG: Handling 1D array (samples only)")
        wav = wav.unsqueeze(0)  # Add channel dimension [1, samples]
        src_channels = 1
    elif wav.ndim == 2:
        # Standard case: [channels, samples]
        src_channels = wav.shape[0]
        print(f"DEBUG: 2D tensor with {src_channels} channels and {wav.shape[1]} samples")
    else:
        # Multi-dimensional case like [batch, channels, samples] or [sources, channels, samples]
        src_channels = wav.shape[-2]  # Assume channel dim is second to last
        print(f"DEBUG: Multi-dim tensor with shape {wav.shape}, src_channels={src_channels}")
    
    # No change needed if already correct number of channels
    if src_channels == channels:
        print("DEBUG: Channel count already correct")
        return wav
        
    # Convert to mono if requested (1 channel)
    if channels == 1:
        if src_channels > 1:
            print("DEBUG: Converting to mono by averaging channels")
            if wav.ndim == 2:
                return wav.mean(dim=0, keepdim=True)  # Average channels: [C, S] -> [1, S]
            else:
                return wav.mean(dim=-2, keepdim=True)  # Keep original dimension structure
        return wav  # Already mono
        
    # Handle mono to multi-channel expansion (upmixing from 1 channel)
    if src_channels == 1:
        print("DEBUG: Expanding mono to multi-channel")
        if wav.ndim == 2:
            # Simple case: [1, samples] -> [channels, samples]
            return wav.expand(channels, wav.shape[1])
        else:
            # More complex case with multi-dimensional tensor
            target_shape = list(wav.shape)
            target_shape[-2] = channels
            return wav.expand(*target_shape)
    
    # Handle downmixing (more channels than needed)
    if src_channels > channels:
        print(f"DEBUG: Downmixing from {src_channels} to {channels} channels")
        if wav.ndim == 2:
            return wav[:channels]  # Take first N channels
        else:
            return wav.narrow(-2, 0, channels)  # Take first N channels in dim=-2
    
    # Handle upmixing from multi-channel to more channels
    print(f"DEBUG: Upmixing from {src_channels} to {channels} channels")
    
    # Process in batches to avoid memory issues
    if wav.ndim == 2:
        # 2D case: [src_channels, samples] -> [channels, samples]
        result = torch.zeros((channels, wav.shape[1]), device=wav.device, dtype=wav.dtype)
        for i in range(channels):
            result[i] = wav[i % src_channels]  # Cyclically map channels
    else:
        # For multi-dimensional tensors, avoid creating the full result tensor at once
        # Instead, create and process it one batch at a time
        result_shape = list(wav.shape)
        result_shape[-2] = channels
        
        print(f"DEBUG: Creating result tensor with shape {result_shape}")
        
        # Check if tensor would be too large (arbitrary threshold of 1GB)
        tensor_size_bytes = torch.tensor([], dtype=wav.dtype).element_size() * np.prod(result_shape)
        if tensor_size_bytes > 1e9:  # 1GB
            print(f"DEBUG: Large tensor detected ({tensor_size_bytes / 1e9:.2f} GB), processing in CPU")
            # Process on CPU to avoid GPU memory issues
            cpu_wav = wav.cpu()
            result = torch.zeros(result_shape, dtype=cpu_wav.dtype)
            for i in range(channels):
                result[..., i, :] = cpu_wav[..., i % src_channels, :]
            return result.to(wav.device)  # Move back to original device
        else:
            result = torch.zeros(result_shape, device=wav.device, dtype=wav.dtype)
            for i in range(channels):
                result[..., i, :] = wav[..., i % src_channels, :]
    
    return result

def convert_audio(wav, from_samplerate, to_samplerate, channels):
    """Convert audio from a given samplerate to a target one and target number of channels."""
    # Convert channels first
    wav = convert_audio_channels(wav, channels)
    
    # Resample audio if necessary
    if from_samplerate != to_samplerate:
        wav = julius.resample_frac(wav, from_samplerate, to_samplerate)
    return wav


# model
def load_model(model, checkpoint_path):
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError("No model checkpoint file found at " + str(checkpoint_path))

        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

        if 'best_state' not in checkpoint:
            raise KeyError("Checkpoint does not contain the state")
            
        state_dict = checkpoint['best_state']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
               new_state_dict[k[7:]] = v
            else:
               new_state_dict[k] = v

        model.load_state_dict(new_state_dict)
        return model

def copy_state(state):
    return {k: v.cpu().clone() for k, v in state.items()}

@contextmanager
def swap_state(model, state):
    """
    Context manager that swaps the state of a model, e.g:

        # model is in old state
        with swap_state(model, new_state):
            # model in new state
        # model back to old state
    """
    old_state = copy_state(model.state_dict())
    model.load_state_dict(state, strict=False)
    try:
        yield
    finally:
        model.load_state_dict(old_state)

#other
@contextmanager
def temp_filenames(count: int, delete=True):
    names = []
    try:
        for _ in range(count):
            names.append(tempfile.NamedTemporaryFile(delete=False).name)
        yield names
    finally:
        if delete:
            for name in names:
                os.unlink(name)

def center_trim(tensor: torch.Tensor, reference: tp.Union[torch.Tensor, int]):
    """
    Center trim `tensor` with respect to `reference`, along the last dimension.
    `reference` can also be a number, representing the length to trim to.
    If the size difference != 0 mod 2, the extra sample is removed on the right side.
    """
    ref_size: int
    if isinstance(reference, torch.Tensor):
        ref_size = reference.size(-1)
    else:
        ref_size = reference
    delta = tensor.size(-1) - ref_size
    if delta < 0:
        raise ValueError("tensor must be larger than reference. " f"Delta is {delta}.")
    if delta:
        tensor = tensor[..., delta // 2:-(delta - delta // 2)]
    return tensor

def EMA(beta: float = 1):
    """
    Exponential Moving Average callback.
    Returns a single function that can be called to repeatidly update the EMA
    with a dict of metrics. The callback will return
    the new averaged dict of metrics.

    Note that for `beta=1`, this is just plain averaging.
    """
    fix: tp.Dict[str, float] = defaultdict(float)
    total: tp.Dict[str, float] = defaultdict(float)

    def _update(metrics: dict, weight: float = 1) -> dict:
        nonlocal total, fix
        for key, value in metrics.items():
            total[key] = total[key] * beta + weight * float(value)
            fix[key] = fix[key] * beta + weight
        return {key: tot / fix[key] for key, tot in total.items()}
    return _update

class DummyPoolExecutor:
    class DummyResult:
        def __init__(self, func, *args, **kwargs):
            self.func = func
            self.args = args
            self.kwargs = kwargs

        def result(self):
            return self.func(*self.args, **self.kwargs)

    def __init__(self, workers=0):
        pass

    def submit(self, func, *args, **kwargs):
        return DummyPoolExecutor.DummyResult(func, *args, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        return

# metric
def new_sdr(references, estimates):
    """
    Compute the SDR for a song
    """
    assert references.dim() == 4
    assert estimates.dim() == 4
    delta = 1e-7  # avoid numerical errors
    num = torch.sum(torch.square(references), dim=(2, 3))
    den = torch.sum(torch.square(references - estimates), dim=(2, 3))
    num += delta
    den += delta
    scores = 10 * torch.log10(num / den)
    return scores



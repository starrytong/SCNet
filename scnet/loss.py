import torch
import torch.nn.functional as F

def spec_rmse_loss(estimate, sources, stft_config):

    _, _, _, lenc = estimate.shape
    spec_estimate = estimate.view(-1, lenc)
    spec_sources = sources.view(-1, lenc)

    spec_estimate = torch.stft(spec_estimate, **stft_config, return_complex=True)
    spec_sources = torch.stft(spec_sources, **stft_config, return_complex=True)


    spec_estimate = torch.view_as_real(spec_estimate)
    spec_sources = torch.view_as_real(spec_sources)

    new_shape = estimate.shape[:-1] + spec_estimate.shape[-3:]
    spec_estimate = spec_estimate.view(*new_shape)
    spec_sources = spec_sources.view(*new_shape)


    loss = F.mse_loss(spec_estimate, spec_sources, reduction='none')


    dims = tuple(range(2, loss.dim()))
    loss = loss.mean(dims).sqrt().mean(dim=(0, 1))  

    return loss


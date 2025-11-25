import itertools
import math
import os
import typing
from dataclasses import dataclass

import hydra.utils
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
import transformers
from torch import Tensor

import classifier
import dataloader
import models
import noise_schedule
import utils

LOG2 = math.log(2)


def _sample_categorical(categorical_probs):
  gumbel_norm = (
    1e-10
    - (torch.rand_like(categorical_probs) + 1e-10).log())
  return (categorical_probs / gumbel_norm).argmax(dim=-1)


def _unsqueeze(x, reference):
  return x.view(
    * x.shape,
    * ((1,) * (len(reference.shape) - len(x.shape))))


@dataclass
class Loss:
  loss: torch.FloatTensor
  nlls: torch.FloatTensor
  token_mask: torch.FloatTensor
  reg_mse: torch.FloatTensor

class MSE(torchmetrics.aggregation.MeanMetric):
  pass


class NLL(torchmetrics.aggregation.MeanMetric):
  pass


class BPD(NLL):
  def compute(self) -> Tensor:
    """Computes the bits per dimension.

    Returns:
      bpd
    """
    return self.mean_value / self.weight / LOG2


class Perplexity(NLL):
  def compute(self) -> Tensor:
    """Computes the Perplexity.

    Returns:
     Perplexity
    """
    return torch.exp(self.mean_value / self.weight)


class Diffusion(L.LightningModule):
  def __init__(
    self,
    config,
    tokenizer: transformers.PreTrainedTokenizer):
    super().__init__()
    self.save_hyperparameters()
    self.config = config

    #print (tokenizer)
    #print ('print tokenizer')

    self.tokenizer = tokenizer
    #self.vocab_size = self.tokenizer.vocab_size
    self.vocab_size = len(self.tokenizer.get_vocab())
    self.sampler = self.config.sampling.predictor
    self.gen_ppl_eval_model_name_or_path = self.config.eval.\
      gen_ppl_eval_model_name_or_path
    self.antithetic_sampling = self.config.training.antithetic_sampling
    self.importance_sampling = self.config.training.importance_sampling
    self.change_of_variables = self.config.training.change_of_variables
    if (not hasattr(self.tokenizer, 'mask_token')
        or self.tokenizer.mask_token is None):
      self.mask_index = self.vocab_size
      self.vocab_size += 1
    else:
      self.mask_index = self.tokenizer.mask_token_id
    self.parameterization = self.config.parameterization
    if self.config.backbone == 'dit':
      self.backbone = models.dit.DIT(
        self.config, vocab_size=self.vocab_size)
    #elif self.config.backbone == 'dimamba':
    #  self.backbone = models.dimamba.DiMamba(
    #    self.config,
    #    vocab_size=self.vocab_size,
    #    pad_token_id=self.tokenizer.pad_token_id)
    elif self.config.backbone == 'ar':
      self.backbone = models.autoregressive.AR(
        self.config,
        vocab_size=self.vocab_size,
        mask_index=self.mask_index)
    elif self.config.backbone == 'hf_dit':
      self.backbone = transformers.AutoModelForMaskedLM.from_pretrained(
        config.eval.checkpoint_path, cache_dir="/data1/fangping/DLcache", trust_remote_code=True)
    else:
      raise ValueError(
        f'Unknown backbone: {self.config.backbone}')

    self.T = self.config.T
    self.subs_masking = self.config.subs_masking

    # Diffusion type configuration
    self.diffusion = self.config.diffusion
    if self.diffusion == 'absorbing_state':
      self.limiting_distribution = None
    elif self.diffusion == 'uniform':
      # For uniform diffusion, use uniform distribution over vocab
      self.limiting_distribution = torch.ones(
        self.vocab_size, device=self.device) / self.vocab_size
    else:
      raise ValueError(f'Unknown diffusion type: {self.diffusion}')

    self.softplus = torch.nn.Softplus()
    # metrics are automatically reset at end of epoch
    metrics = torchmetrics.MetricCollection({
      'nll': NLL(),
      'bpd': BPD(),
      'ppl': Perplexity(),
    })
    metrics.set_dtype(torch.float64)
    self.train_metrics = metrics.clone(prefix='train/')
    self.valid_metrics = metrics.clone(prefix='val/')
    self.test_metrics = metrics.clone(prefix='test/')


    metrics_mse = torchmetrics.MetricCollection({
      'reg_mse': MSE(),
    })
    metrics_mse.set_dtype(torch.float64)
    self.train_metrics_mse = metrics_mse.clone(prefix='train_MSE/')
    self.valid_metrics_mse = metrics_mse.clone(prefix='val_MSE/')
    self.test_metrics_mse = metrics_mse.clone(prefix='test_MSE/')

    # generative perplexity
    self.gen_ppl_metric = Perplexity()
    self.eval_model_tokenizer = transformers.AutoTokenizer.\
      from_pretrained(self.gen_ppl_eval_model_name_or_path, cache_dir="/data1/fangping/DLcache")
    if self.eval_model_tokenizer.pad_token is None:
      self.eval_model_tokenizer.pad_token =\
          self.eval_model_tokenizer.eos_token
      self.eval_model_tokenizer.pad_token_id =\
          self.eval_model_tokenizer.eos_token_id

    self.noise = noise_schedule.get_noise(self.config,
                                          dtype=self.dtype)
    if self.config.training.ema > 0:
      self.ema = models.ema.ExponentialMovingAverage(
        itertools.chain(self.backbone.parameters(),
                        self.noise.parameters()),
        decay=self.config.training.ema)
    else:
      self.ema = None
    
    self.lr = self.config.optim.lr
    self.sampling_eps = self.config.training.sampling_eps
    self.time_conditioning = self.config.time_conditioning
    self.neg_infinity = -1000000.0
    self.fast_forward_epochs = None
    self.fast_forward_batches = None
    self._validate_configuration()

  def _validate_configuration(self):
    assert not (self.change_of_variables
                and self.importance_sampling)
    if self.parameterization == 'sedd':
      assert not self.importance_sampling
      assert not self.change_of_variables
    if self.parameterization == 'd3pm':
      assert self.T > 0
    if self.T > 0:
      assert self.parameterization in {'d3pm', 'subs'}
    if self.subs_masking:
      assert self.parameterization == 'd3pm'

  def on_load_checkpoint(self, checkpoint):
    if self.ema:
      self.ema.load_state_dict(checkpoint['ema'])
    # Copied from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py#L41
    self.fast_forward_epochs = checkpoint['loops'][
      'fit_loop']['epoch_progress']['current']['completed']
    self.fast_forward_batches = checkpoint['loops'][
      'fit_loop']['epoch_loop.batch_progress'][
        'current']['completed']

  def on_save_checkpoint(self, checkpoint):
    if self.ema:
      checkpoint['ema'] = self.ema.state_dict()
    # Copied from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/tasks/seq.py
    # ['epoch_loop.batch_progress']['total']['completed'] is 1 iteration
    # behind, so we're using the optimizer's progress.
    checkpoint['loops']['fit_loop'][
      'epoch_loop.batch_progress']['total'][
        'completed'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['total'][
              'completed'] * self.trainer.accumulate_grad_batches
    checkpoint['loops']['fit_loop'][
      'epoch_loop.batch_progress']['current'][
        'completed'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['current'][
              'completed'] * self.trainer.accumulate_grad_batches
    # _batches_that_stepped tracks the number of global steps, not the number
    # of local steps, so we don't multiply with self.trainer.accumulate_grad_batches here.
    checkpoint['loops']['fit_loop'][
      'epoch_loop.state_dict'][
        '_batches_that_stepped'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['total']['completed']
    if 'sampler' not in checkpoint.keys():
      checkpoint['sampler'] = {}
    if hasattr(self.trainer.train_dataloader.sampler,
               'state_dict'):
      sampler_state_dict = self.trainer.\
        train_dataloader.sampler.state_dict()
      checkpoint['sampler'][
        'random_state'] = sampler_state_dict.get(
          'random_state', None)
    else:
      checkpoint['sampler']['random_state'] = None

  def on_train_start(self):
    if self.ema:
      self.ema.move_shadow_params_to_device(self.device)
    # Adapted from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
    distributed = (
      self.trainer._accelerator_connector.use_distributed_sampler
      and self.trainer._accelerator_connector.is_distributed)
    if distributed:
      sampler_cls = dataloader.FaultTolerantDistributedSampler
    else:
      sampler_cls = dataloader.RandomFaultTolerantSampler
    updated_dls = []
    for dl in self.trainer.fit_loop._combined_loader.flattened:
      if hasattr(dl.sampler, 'shuffle'):
        dl_sampler = sampler_cls(
          dl.dataset, shuffle=dl.sampler.shuffle)
      else:
        dl_sampler = sampler_cls(dl.dataset)
      if (distributed
          and self.fast_forward_epochs is not None
          and self.fast_forward_batches is not None):
        dl_sampler.load_state_dict({
          'epoch': self.fast_forward_epochs,
          'counter': (self.fast_forward_batches
                      * self.config.loader.batch_size)})
      updated_dls.append(
        torch.utils.data.DataLoader(
          dl.dataset,
          batch_size=self.config.loader.batch_size,
          num_workers=self.config.loader.num_workers,
          pin_memory=self.config.loader.pin_memory,
          sampler=dl_sampler,
          shuffle=False,
          persistent_workers=True))
    self.trainer.fit_loop._combined_loader.flattened = updated_dls

  def optimizer_step(self, *args, **kwargs):
    super().optimizer_step(*args, **kwargs)
    if self.ema:
      self.ema.update(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))

  def _subs_parameterization(self, logits, xt):
    # log prob at the mask index = - infinity
    logits[:, :, self.mask_index] += self.neg_infinity
    
    # Normalize the logits such that x.exp() is
    # a probability distribution over vocab_size.
    logits = logits - torch.logsumexp(logits, dim=-1,
                                      keepdim=True)

    # Apply updates directly in the logits matrix.
    # For the logits of the unmasked tokens, set all values
    # to -infinity except for the indices corresponding to
    # the unmasked tokens.
    unmasked_indices = (xt != self.mask_index)
    logits[unmasked_indices] = self.neg_infinity
    logits[unmasked_indices, xt[unmasked_indices]] = 0
    return logits

  def _d3pm_parameterization(self, logits):
    if self.subs_masking:
      logits[:, :, self.mask_index] += self.neg_infinity
    logits = logits - torch.logsumexp(logits, dim=-1,
                                      keepdim=True)
    return logits

  def _sedd_parameterization(self, logits, xt, sigma):
    esigm1_log = torch.where(
      sigma < 0.5,
      torch.expm1(sigma),
      sigma.exp() - 1).log().to(logits.dtype)
    # logits shape
    # (batch_size, diffusion_model_input_length, vocab_size)
    logits = logits - esigm1_log[:, None, None] - np.log(
      logits.shape[-1] - 1)
    # The below scatter operation sets the log score
    # for the input word to 0.
    logits = torch.scatter(logits, -1, xt[..., None],
                           torch.zeros_like(logits[..., :1]))
    return logits

  def _process_sigma(self, sigma):
    if sigma is None:
      assert self.parameterization == 'ar'
      return sigma
    if sigma.ndim > 1:
      sigma = sigma.squeeze(-1)
    if not self.time_conditioning:
      sigma = torch.zeros_like(sigma)
    assert sigma.ndim == 1, sigma.shape
    return sigma

  def forward(self, x, sigma, regress=False):
    """Returns log score."""
    sigma = self._process_sigma(sigma)
    with torch.cuda.amp.autocast(dtype=torch.float32):
      logits = self.backbone(x, sigma, regress)
    
    if self.parameterization == 'subs':
      return self._subs_parameterization(logits=logits,
                                         xt=x)
    elif self.parameterization == 'sedd':
      return self._sedd_parameterization(logits=logits,
                                         xt=x,
                                         sigma=sigma)
    elif self.parameterization == 'd3pm':
      return self._d3pm_parameterization(logits=logits)
    return logits

  def regression_forward(self, x, sigma, regress=True):
    sigma = self._process_sigma(sigma)
    with torch.cuda.amp.autocast(dtype=torch.float32):
        logits, regression_outs = self.backbone(x, sigma, regress)
    return regression_outs

  def _d3pm_loss(self, model_output, xt, x0, t):
    dt = 1 / self.T

    if torch.is_tensor(t):
      t = t[:, None]
      assert t.ndim == 2
      t = t.clamp(0., 1. - 1e-4)
    alpha_t = 1 - t + torch.zeros_like(xt)
    alpha_s = 1 - (t - dt) + torch.zeros_like(xt)

    log_x_theta_at_x0 = torch.gather(
      model_output, -1, x0[:, :, None]).squeeze(-1)
    log_x_theta_at_m = model_output[:, :, self.mask_index]
    x_theta_at_m = log_x_theta_at_m.exp()
    
    term_1_coef = dt / t
    term_1_log_nr = torch.log(alpha_t * x_theta_at_m / t + 1)
    term_1_log_dr = log_x_theta_at_x0
    
    term_2_coef = 1 - dt / t
    term_2_log_nr = term_1_log_nr
    term_2_log_dr = torch.log(alpha_s * x_theta_at_m / (t - dt) + 1)

    L_vb_masked = (
      term_1_coef * (term_1_log_nr - term_1_log_dr)
      + term_2_coef * (term_2_log_nr - term_2_log_dr))

    L_vb = L_vb_masked * (xt == self.mask_index)

    return self.T * L_vb

  def _compute_loss(self, batch, prefix):
    if 'attention_mask' in batch:
      attention_mask = batch['attention_mask'].to(torch.float32)
    else:
      attention_mask = None
    
    property_labels = batch['descriptors'].to(torch.float32)
    property_labels = (property_labels - self.backbone.regression.mean_stats.to(device=property_labels.device)) / self.backbone.regression.std_stats.to(device=property_labels.device)

    losses = self._loss(batch['input_ids'].to(torch.int64), attention_mask, property_labels)
    loss = losses.loss
    reg_mse = losses.reg_mse

    if prefix == 'train':
      self.train_metrics.update(losses.nlls, losses.token_mask)
      metrics = self.train_metrics
      self.train_metrics_mse.update(reg_mse)
      metrics_reg = self.train_metrics_mse
    elif prefix == 'val':
      self.valid_metrics.update(losses.nlls, losses.token_mask)
      metrics = self.valid_metrics
      self.valid_metrics_mse.update(reg_mse)
      metrics_reg = self.valid_metrics_mse
    elif prefix == 'test':
      self.test_metrics.update(losses.nlls, losses.token_mask)
      metrics = self.test_metrics
      self.test_metrics_mse.update(reg_mse)
      metrics_reg = self.test_metrics_mse
    else:
      raise ValueError(f'Invalid prefix: {prefix}')

    self.log_dict(metrics,
                  on_step=False,
                  on_epoch=True,
                  sync_dist=True)

    self.log_dict(metrics_reg,
                  on_step=False,
                  on_epoch=True,
                  sync_dist=True)

    return loss + 0.1*reg_mse

  def on_train_epoch_start(self):
    self.backbone.train()
    self.noise.train()

  def training_step(self, batch, batch_idx):
    loss = self._compute_loss(batch, prefix='train')
    self.log(name='trainer/loss',
             value=loss.item(),
             on_step=True,
             on_epoch=False,
             sync_dist=True)
    return loss

  def on_validation_epoch_start(self):
    if self.ema:
      self.ema.store(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
      self.ema.copy_to(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
    self.backbone.eval()
    self.noise.eval()
    assert self.valid_metrics.nll.mean_value == 0
    assert self.valid_metrics.nll.weight == 0

  def validation_step(self, batch, batch_idx):
    return self._compute_loss(batch, prefix='val')

  def on_validation_epoch_end(self):
    if ((self.config.eval.compute_perplexity_on_sanity
         or not self.trainer.sanity_checking)
         and self.config.eval.generate_samples
         and not self.parameterization == 'ar'):
      # TODO(justin): implement sampling and kv cache for AR
      samples, text_samples = None, None
      for _ in range(
        self.config.sampling.num_sample_batches):
        samples = self._sample()
        # Decode the samples to be re-tokenized by eval model
        text_samples = self.tokenizer.batch_decode(samples)
        if self.config.eval.compute_generative_perplexity:
          self.compute_generative_perplexity(text_samples)
      if self.trainer.global_rank == 0 and hasattr(
        self.trainer.logger, 'log_table'):
        # Log the last generated samples
        text_samples = text_samples[
          : self.config.sampling.num_sample_log]
        self.trainer.logger.log_table(
          key=f'samples@global_step{self.global_step}',
          columns=['Generated Samples'],
          data=[[s] for s in text_samples])
      if self.config.eval.compute_generative_perplexity:
        self.log('val/gen_ppl',
                 self.gen_ppl_metric,
                 on_epoch=True,
                 on_step=False,
                 sync_dist=True)
    if self.ema:
      self.ema.restore(
        itertools.chain(self.backbone.parameters(),
                        self.noise.parameters()))

  def configure_optimizers(self):
    # TODO(yair): Lightning currently giving this warning when using `fp16`:
    #  "Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
    #  Not clear if this is a problem or not.
    #  See: https://github.com/Lightning-AI/pytorch-lightning/issues/5558
    optimizer = torch.optim.AdamW(
      itertools.chain(self.backbone.parameters(),
                      self.noise.parameters()),
      lr=self.config.optim.lr,
      betas=(self.config.optim.beta1,
             self.config.optim.beta2),
      eps=self.config.optim.eps,
      weight_decay=self.config.optim.weight_decay)

    scheduler = hydra.utils.instantiate(
      self.config.lr_scheduler, optimizer=optimizer)
    scheduler_dict = {
      'scheduler': scheduler,
      'interval': 'step',
      'monitor': 'val/loss',
      'name': 'trainer/lr',
    }
    return [optimizer], [scheduler_dict]

  @torch.no_grad()
  def eval_retokenize(self, text_samples, max_length):
    """Retokenizes samples for the eval model.
    
    Args:
        text_samples: List of sentences generated by the model.
    Returns:
        samples: Samples re-tokenized for the eval model
        attn_mask: Attention mask for the eval model
        eval_context_size: Size of the context for the eval model
    """
    if 'llama2' in self.gen_ppl_eval_model_name_or_path:
      tokenizer_kwargs = {
        'text_samples': text_samples,
        'return_tensors': 'pt',
        'return_token_type_ids': False,
        'return_attention_mask': True,
        'truncation': True,
        'padding': True,
        'max_length': max_length,
      }
      eval_context_size = 4096
    else:
      tokenizer_kwargs = {
        'return_tensors': 'pt',
        'return_token_type_ids': False,
        'return_attention_mask': True,
        'truncation': True,
        'padding': True,
        'max_length': max_length,
      }
      eval_context_size = 1024
    samples = self.eval_model_tokenizer(
      text_samples, ** tokenizer_kwargs)
    attn_mask = samples['attention_mask']
    samples = samples['input_ids']
    if 'llama2' not in self.gen_ppl_eval_model_name_or_path:
      attn_mask = attn_mask.to(self.device)
      samples = samples.to(self.device)      
    return samples, attn_mask, eval_context_size

  @torch.no_grad()
  def compute_generative_perplexity(
    self,
    text_samples: typing.List[str],
    retokenize: bool = True,
    max_length: typing.Optional[int] = None) -> None:
    """Compute the generative perplexity of the model.

    Args:
        text_samples: List of sentences generated by the model.
    
    Returns:
        Perplexity of the generated text under a different
        pre-trained AR model (e.g., GPT2).
    """
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    eval_model = transformers.AutoModelForCausalLM.from_pretrained(
      self.gen_ppl_eval_model_name_or_path, cache_dir="/data1/fangping/DLcache").eval()
    if max_length is None:
      max_length = self.config.model.length
    if 'llama2' not in self.gen_ppl_eval_model_name_or_path:
      eval_model = eval_model.to(self.device)
    # Re-tokenize using eval model's tokenizer
    if retokenize:
      (samples, attn_mask,
       eval_context_size) = self.eval_retokenize(
         text_samples, max_length=max_length)
    else:
      samples = text_samples
      attn_mask = torch.ones(samples.shape).to(self.device)
      eval_context_size = samples.shape[-1]
    batch_size = min(
      self.config.eval.perplexity_batch_size,
      samples.shape[0])
    num_batches = samples.shape[0] // batch_size
    for i in range(num_batches):
      _samples = torch.split(
        samples[i * batch_size: (i + 1) * batch_size],
        eval_context_size,
        dim=-1)
      _attn_mask = torch.split(
        attn_mask[i * batch_size: (i + 1) * batch_size],
        eval_context_size,
        dim=-1)
      for (sample_chunk, attn_mask_chunk) in zip(
        _samples, _attn_mask):
        logits = eval_model(
          sample_chunk, attention_mask=attn_mask_chunk)[0]
        logits = logits.transpose(-1, -2)
        
        nlls = F.cross_entropy(logits[..., :-1],
                               sample_chunk[..., 1:],
                               reduction='none')
        first_eos = (sample_chunk == self.eval_model_tokenizer\
                     .eos_token_id).cumsum(-1) == 1
        token_mask = (
          sample_chunk
          != self.eval_model_tokenizer.eos_token_id)
        self.gen_ppl_metric.update(
          nlls, first_eos[..., 1:] + token_mask[..., 1:])

  def q_xt(self, x, move_chance):
    """Computes the noisy sample xt.

    Args:
      x: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input. 
      move_chance: float torch.Tensor with shape (batch_size, 1).
    """
    move_indices = torch.rand(
      * x.shape, device=x.device) < move_chance
    xt = torch.where(move_indices, self.mask_index, x)
    return xt

  def _sample_prior(self, *batch_dims):
    return self.mask_index * torch.ones(
      * batch_dims, dtype=torch.int64)

  def _compute_posterior(self, x, xt, alpha_s, alpha_t):
    """Computes the posterior / approximate posterior.

    Args:
      x: Either clean input `x0` (one-hot),
        or model's predicted `x_theta` of shape (B, L, V).
      xt: The noisy latent (as indices) of shape (B, L).
      alpha_s: Noise level at s of shape (B, [L | 1], 1).
      alpha_t: Noise level at t of shape (B, [L | 1], 1).

    Returns:
      Posterior / approximate posterior of shape (B, L, V).
    """
    alpha_ts = alpha_t / alpha_s
    d_alpha = alpha_s - alpha_t
    xt_one_hot = F.one_hot(xt, self.vocab_size)
    if self.diffusion == 'uniform':
      return (
        (alpha_t * self.vocab_size * x * xt_one_hot +
         (alpha_ts - alpha_t) * xt_one_hot +
         d_alpha * x +
         (1 - alpha_ts) * (1 - alpha_s) * self.limiting_distribution)
        /
        (alpha_t * self.vocab_size * torch.gather(x, -1, xt[..., None]) +
         (1 - alpha_t))
      )
    raise NotImplementedError(
      f"Diffusion type {self.diffusion} not implemented.")

  def _ddpm_caching_update(self, x, t, dt, p_x0=None):
    assert self.config.noise.type == 'loglinear'
    sigma_t, _ = self.noise(t)
    if t.ndim > 1:
      t = t.squeeze(-1)
    assert t.ndim == 1
    move_chance_t = t[:, None, None]
    move_chance_s = (t - dt)[:, None, None]
    assert move_chance_t.ndim == 3, move_chance_t.shape
    if p_x0 is None:
      p_x0 = self.forward(x, sigma_t).exp()
    
    assert move_chance_t.ndim == p_x0.ndim
    q_xs = p_x0 * (move_chance_t - move_chance_s)
    q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
    _x = _sample_categorical(q_xs)
    
    copy_flag = (x != self.mask_index).to(x.dtype)
    return p_x0, copy_flag * x + (1 - copy_flag) * _x

  def _ddpm_update(self, x, t, dt):
    sigma_t, _ = self.noise(t)
    sigma_s, _ = self.noise(t - dt)
    if sigma_t.ndim > 1:
      sigma_t = sigma_t.squeeze(-1)
    if sigma_s.ndim > 1:
      sigma_s = sigma_s.squeeze(-1)
    assert sigma_t.ndim == 1, sigma_t.shape
    assert sigma_s.ndim == 1, sigma_s.shape
    move_chance_t = 1 - torch.exp(-sigma_t)
    move_chance_s = 1 - torch.exp(-sigma_s)
    move_chance_t = move_chance_t[:, None, None]
    move_chance_s = move_chance_s[:, None, None]
    unet_conditioning = sigma_t
    log_p_x0 = self.forward(x, unet_conditioning)
    assert move_chance_t.ndim == log_p_x0.ndim
    # Technically, this isn't q_xs since there's a division
    # term that is missing. This division term doesn't affect
    # the samples.
    q_xs = log_p_x0.exp() * (move_chance_t
                             - move_chance_s)
    q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
    _x = _sample_categorical(q_xs)

    copy_flag = (x != self.mask_index).to(x.dtype)
    return copy_flag * x + (1 - copy_flag) * _x

  def _ar_sampler(self, bsz):
    # precompute token buffer
    num_pred_tokens = self.config.model.length - 1
    x = torch.zeros(
      (bsz, num_pred_tokens + 1),
      dtype=torch.long,
      device=self.device)
    x[:, 0] = self.tokenizer.bos_token_id
    # precompute noise
    noise = (torch.distributions.Gumbel(0, 1)
             .sample((bsz, num_pred_tokens, self.vocab_size))
             .to(self.device))
    for i in range(num_pred_tokens):
      next_logits = self.forward(x[:, :i + 1], None)[:, -1]
      y = (next_logits + noise[:, i]).argmax(-1)
      x[:, i + 1] = y
    return x

  @torch.no_grad()
  def _sample(self, num_steps=None, eps=1e-5):
    """Generate samples from the model."""
    batch_size_per_gpu = self.config.loader.eval_batch_size
    if self.parameterization == 'ar':
      return self._ar_sampler(batch_size_per_gpu)
    # Lightning auto-casting is not working in this method for some reason
    if num_steps is None:
      num_steps = self.config.sampling.steps
    x = self._sample_prior(
      batch_size_per_gpu,
      self.config.model.length).to(self.device)
    timesteps = torch.linspace(
      1, eps, num_steps + 1, device=self.device)
    dt = (1 - eps) / num_steps
    p_x0_cache = None

    for i in range(num_steps):
      t = timesteps[i] * torch.ones(
        x.shape[0], 1, device=self.device)
      if self.sampler == 'ddpm':
        x = self._ddpm_update(x, t, dt)
      elif self.sampler == 'ddpm_cache':
        p_x0_cache, x_next = self._ddpm_caching_update(
          x, t, dt, p_x0=p_x0_cache)
        if (not torch.allclose(x_next, x)
            or self.time_conditioning):
          # Disable caching
          p_x0_cache = None
        x = x_next
      else:
        x = self._analytic_update(x, t, dt)

    if self.config.sampling.noise_removal:
      t = timesteps[-1] * torch.ones(x.shape[0], 1,
                                     device=self.device)
      if self.sampler == 'analytic':
        x = self._denoiser_update(x, t)
      else:
        unet_conditioning = self.noise(t)[0]
        x = self.forward(x, unet_conditioning).argmax(dim=-1)
    return x

  def restore_model_and_sample(self, num_steps, eps=1e-5):
    """Generate samples from the model."""
    # Lightning auto-casting is not working in this method for some reason
    if self.ema:
      self.ema.store(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
      self.ema.copy_to(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
    self.backbone.eval()
    self.noise.eval()
    samples = self._sample(num_steps=num_steps, eps=eps)
    if self.ema:
      self.ema.restore(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
    self.backbone.train()
    self.noise.train()
    return samples

  def get_score(self, x, sigma):
    model_output = self.forward(x, sigma)
    if self.parameterization == 'subs':
      # score(x, t) = p_t(y) / p_t(x)
      # => log score(x, t) = log p_t(y) - log p_t(x)
      
      # case 1: x = masked
      #   (i) y = unmasked
      #     log score(x, t) = log p_\theta(x)|_y + log k
      #     where k = exp(- sigma) / (1 - exp(- sigma))
      #   (ii) y = masked
      #     log score(x, t) = 0

      # case 2: x = unmasked
      #   (i) y != masked, y != x
      #     log score(x_i, t) = - inf
      #   (ii) y = x 
      #     log score(x_i, t) = 0
      #   (iii) y = masked token
      #     log score(x_i, t) = - log k
      #     where k = exp(- sigma) / (1 - exp(- sigma))
      
      log_k = - torch.log(torch.expm1(sigma)).squeeze(-1)
      assert log_k.ndim == 1
      
      masked_score = model_output + log_k[:, None, None]
      masked_score[:, :, self.mask_index] = 0

      unmasked_score = self.neg_infinity * torch.ones_like(
        model_output)
      unmasked_score = torch.scatter(
        unmasked_score,
        -1,
        x[..., None],
        torch.zeros_like(unmasked_score[..., :1]))
      unmasked_score[:, :, self.mask_index] = - (
        log_k[:, None] * torch.ones_like(x))
      
      masked_indices = (x == self.mask_index).to(
        model_output.dtype)[:, :, None]
      model_output = (
        masked_score * masked_indices
        + unmasked_score * (1 - masked_indices))
    return model_output.exp()

  def _staggered_score(self, score, dsigma):
    score = score.clone()
    extra_const = (1 - dsigma.exp()) * score.sum(dim=-1)
    score *= dsigma.exp()[:, None]
    score[..., self.mask_index] += extra_const
    return score

  def _analytic_update(self, x, t, step_size):
    curr_sigma, _ = self.noise(t)
    next_sigma, _ = self.noise(t - step_size)
    dsigma = curr_sigma - next_sigma
    score = self.get_score(x, curr_sigma)
    stag_score = self._staggered_score(score, dsigma)
    probs = stag_score * self._transp_transition(x, dsigma)
    return _sample_categorical(probs)

  def _denoiser_update(self, x, t):
    sigma, _ = self.noise(t)
    score = self.get_score(x, sigma)
    stag_score = self._staggered_score(score, sigma)
    probs = stag_score * self._transp_transition(x, sigma)
    probs[..., self.mask_index] = 0
    samples = _sample_categorical(probs)
    return samples

  def _transp_transition(self, i, sigma):
    sigma = _unsqueeze(sigma, reference=i[..., None])
    edge = torch.exp(-sigma) * F.one_hot(
      i, num_classes=self.vocab_size)
    edge += torch.where(i == self.mask_index,
                        1 - torch.exp(-sigma).squeeze(-1),
                        0)[..., None]
    return edge

  def _sample_t(self, n, device):
    _eps_t = torch.rand(n, device=device)
    if self.antithetic_sampling:
      offset = torch.arange(n, device=device) / n
      _eps_t = (_eps_t / n + offset) % 1
    t = (1 - self.sampling_eps) * _eps_t + self.sampling_eps
    if self.importance_sampling:
      return self.noise.importance_sampling_transformation(t)
    return t

  def _maybe_sub_sample(self, x0, attention_mask):
    seqlen = x0.shape[1]
    if seqlen > self.config.model.length:
      assert seqlen == 2 * self.config.model.length
      # cropping is needed for text8-crop dataset
      # try the same starting point for now
      start = np.random.choice(self.config.model.length)
      end = start + self.config.model.length
      input_tokens = x0[:, start: end]
      output_tokens = x0[:, start + 1: end + 1]
      new_attention_mask = attention_mask[:, start: end]

      # Helps with validation PPL, since the val
      # examples will all start and end with BOS/EOS
      input_tokens[:, 0] = self.tokenizer.bos_token_id
      output_tokens[:, -1] = self.tokenizer.eos_token_id
    elif self.parameterization == 'ar':
      input_tokens = x0[:, :-1]
      output_tokens = x0[:, 1:]
      new_attention_mask = attention_mask[:, 1:]
    else:
      input_tokens = x0
      output_tokens = None
      new_attention_mask = attention_mask
    return input_tokens, output_tokens, new_attention_mask

  def _reconstruction_loss(self, x0):
    t0 = torch.zeros(x0.shape[0], dtype=self.dtype,
                     device=self.device)
    assert self.config.noise.type == 'loglinear'
    # The above assert is for d3pm parameterization
    unet_conditioning = self.noise(t0)[0][:, None]
    model_output_t0 = self.forward(x0, unet_conditioning)
    return - torch.gather(input=model_output_t0,
                          dim=-1,
                          index=x0[:, :, None]).squeeze(-1)

  def _forward_pass_diffusion(self, x0):
    t = self._sample_t(x0.shape[0], x0.device)
    if self.T > 0:
      t = (t * self.T).to(torch.int)
      t = t / self.T
      # t \in {1/T, 2/T, ..., 1}
      t += (1 / self.T)

    if self.change_of_variables:
      unet_conditioning = t[:, None]
      f_T = torch.log1p(- torch.exp(- self.noise.sigma_max))
      f_0 = torch.log1p(- torch.exp(- self.noise.sigma_min))
      move_chance = torch.exp(f_0 + t * (f_T - f_0))
      move_chance = move_chance[:, None]
    else:
      sigma, dsigma = self.noise(t)
      unet_conditioning = sigma[:, None]
      move_chance = 1 - torch.exp(-sigma[:, None])

    xt = self.q_xt(x0, move_chance)
    model_output = self.forward(xt, unet_conditioning)
    utils.print_nans(model_output, 'model_output')

    if self.parameterization == 'sedd':
      return dsigma[:, None] * self._score_entropy(
        model_output, sigma[:, None], xt, x0)
    
    if self.T > 0:
      diffusion_loss = self._d3pm_loss(
        model_output=model_output, xt=xt, x0=x0, t=t)
      if self.parameterization == 'd3pm':
        reconstruction_loss = self._reconstruction_loss(x0)
      elif self.parameterization == 'subs':
        reconstruction_loss = 0
      return reconstruction_loss + diffusion_loss
    
    # SUBS parameterization, continuous time.
    log_p_theta = torch.gather(
      input=model_output,
      dim=-1,
      index=x0[:, :, None]).squeeze(-1)
    
    if self.change_of_variables or self.importance_sampling:
      return log_p_theta * torch.log1p(
        - torch.exp(- self.noise.sigma_min))
    
    return - log_p_theta * (
      dsigma / torch.expm1(sigma))[:, None]

  def _loss(self, x0, attention_mask, property_labels):
    (input_tokens, output_tokens,
     attention_mask) = self._maybe_sub_sample(
       x0, attention_mask)

    if self.parameterization == 'ar':
      logprobs = self.backbone(input_tokens, None)
      loss = - logprobs.gather(
        -1, output_tokens[:, :, None])[:, :, 0]
    else:
      loss = self._forward_pass_diffusion(input_tokens)
    
    nlls = loss * attention_mask
    count = attention_mask.sum()

    batch_nll = nlls.sum()
    token_nll = batch_nll / count

    t_reg = 0 * self._sample_t(x0.shape[0], x0.device) + 1e-3
    sigma0,_ = self.noise(t_reg)
    unet_conditioning0 = sigma0[:, None]
    move_chance_impossible = -1
    x_complete = self.q_xt(x0, move_chance_impossible)
    property_outs = self.regression_forward(x_complete, unet_conditioning0)

    reg_mse = torch.mean((property_outs-property_labels)**2)

    return Loss(loss=token_nll,
                nlls=nlls,
                token_mask=attention_mask,
                reg_mse=reg_mse)

  def _score_entropy(self, log_score, sigma, xt, x0):
    """Computes the SEDD loss.

    Args:
      log_score: float torch.Tensor with shape (batch_size,
          diffusion_model_input_length, vocab_size),
          log score, output of the denoising network.
      xt: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input.
      x0: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input.
      sigma: float torch.Tensor with shape (batch_size, 1).

    Returns:
      loss with shape (batch_size, diffusion_model_input_length)
    """
    masked_indices = xt == self.mask_index

    expsig_minus_1 = torch.expm1(sigma).expand_as(xt)
    q_ratio = 1 / expsig_minus_1[masked_indices]

    words_that_were_masked = x0[masked_indices]

    neg_term = q_ratio * torch.gather(
      log_score[masked_indices],
      -1,
      words_that_were_masked[..., None]).squeeze(-1)
    score = log_score[masked_indices].exp()
    if self.mask_index == self.vocab_size - 1:
      pos_term = score[:, :-1].sum(dim=-1)
    else:
      pos_term = score[:, : self.mask_index].sum(
        dim=-1) + score[:, self.mask_index + 1:].sum(dim=-1)
    const = q_ratio * (q_ratio.log() - 1)

    entropy = torch.zeros(* xt.shape, device=xt.device)
    entropy[masked_indices] += pos_term - neg_term + const
    return entropy

  @torch.no_grad
  def sample_subs_guidance(
    self, n_samples, stride_length, num_strides, dt=0.001):
    ones = torch.ones(n_samples, dtype=self.dtype,
                      device=self.device)

    num_steps = int(1 / dt)
    sampling_steps = 0
    intermediate_tokens = []
    target = None
    for _ in range(num_strides + 1):
      p_x0_cache = None
      x = self._sample_prior(
        n_samples,
        self.config.model.length).to(self.device)
      if target is not None:
        x[:, : -stride_length] = target
      for i in range(num_steps + 1):
        p_x0_cache, x_next = self._ddpm_caching_update(
          x=x, t=(1 - i * dt) * ones, dt=dt, p_x0=p_x0_cache)
        if (not torch.allclose(x_next, x)
            or self.time_conditioning):
          p_x0_cache = None
          sampling_steps += 1
        x = x_next
      x = self.forward(x, 0 * ones).argmax(dim=-1)
      intermediate_tokens.append(
        x[:, :stride_length].cpu().numpy())
      target = x[:, stride_length:]
    
    intermediate_tokens.append(target.cpu().numpy())
    intermediate_text_samples = []
    sequence_lengths = ((
      np.concatenate(intermediate_tokens, axis=1)[:, 1:]
      == self.tokenizer.eos_token_id).cumsum(-1) == 0).sum(-1)
    for i in range(2, len(intermediate_tokens) + 1):
      intermediate_text_samples.append(
        self.tokenizer.batch_decode(
          np.concatenate(intermediate_tokens[:i], axis=1)))
    return (sampling_steps, intermediate_text_samples,
            sequence_lengths)

  def restore_model_and_semi_ar_sample(
      self, stride_length, num_strides, dt=0.001):
    """Generate samples from the model."""
    # Lightning auto-casting is not working in this method for some reason
    if self.ema:
      self.ema.store(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
      self.ema.copy_to(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
    self.backbone.eval()
    self.noise.eval()
    (sampling_steps, samples,
     sequence_lengths) = self.sample_subs_guidance(
      n_samples=self.config.loader.eval_batch_size,
      stride_length=stride_length,
      num_strides=num_strides, 
      dt=dt)
    if self.ema:
      self.ema.restore(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
    self.backbone.train()
    self.noise.train()
    return sampling_steps, samples, sequence_lengths

  # ========================================================================
  # Classifier-Based Guided Generation Methods
  # ========================================================================

  def _diffusion_sample(
    self,
    classifier_model: typing.Optional[classifier.Classifier] = None,
    cond: typing.Optional[torch.tensor] = None,
    eps: float = 1e-5,
  ):
    """Main sampling loop with support for guided generation.

    Iteratively denoises from pure noise to generate samples, with optional
    classifier-based guidance. Supports multiple guidance methods configured
    via self.config.guidance.method.

    Args:
      classifier_model: Optional classifier for guided generation
      cond: Optional conditioning for CFG
      eps: Minimum timestep (default: 1e-5)

    Returns:
      Generated samples as token indices of shape (batch_size, seq_len)
    """
    # Sample from prior (all mask tokens for absorbing state)
    # Use eval_batch_size to match existing code convention
    batch_size = getattr(self.config.sampling, 'batch_size',
                         self.config.loader.eval_batch_size)
    xt = self._sample_prior(
      batch_size,
      self.config.model.length
    ).to(self.device)

    # Create timestep schedule
    timesteps = torch.linspace(
      1, eps, self.config.sampling.steps + 1, device=self.device)
    dt = (1 - eps) / self.config.sampling.steps

    # Sampling loop with progress bar
    try:
      from tqdm import tqdm
      pbar = tqdm(range(self.config.sampling.steps), desc='Sampling', leave=False)
    except ImportError:
      pbar = range(self.config.sampling.steps)

    NFEs = 0  # Number of function evaluations
    cache = None

    for i in pbar:
      t = timesteps[i]

      # Handle discrete time if T > 0
      if self.T > 0:
        t = (t * self.T).to(torch.int)
        t = t / self.T
        t += (1 / self.T)

      t = t * torch.ones(xt.shape[0], 1, device=self.device)

      if cache is None:
        NFEs += 1

      # Compute noise levels
      sigma_t, _ = self.noise(t)
      sigma_s, _ = self.noise(t - dt)

      # Ensure proper dimensions
      if sigma_t.ndim > 1:
        sigma_t = sigma_t.squeeze(-1)
      if sigma_s.ndim > 1:
        sigma_s = sigma_s.squeeze(-1)
      assert sigma_t.ndim == 1, sigma_t.shape
      assert sigma_s.ndim == 1, sigma_s.shape

      # Compute corruption probabilities
      move_chance_t = 1 - torch.exp(-sigma_t)
      move_chance_s = 1 - torch.exp(-sigma_s)
      move_chance_t = move_chance_t[:, None, None]
      move_chance_s = move_chance_s[:, None, None]
      assert move_chance_t.ndim == 3, move_chance_t.shape

      # Route to appropriate denoising method
      if getattr(self.config, 'guidance', None) is None:
        # No guidance - standard DDPM
        xs, q_xs, cache = self._ddpm_denoise(
          xt=xt,
          time_conditioning=sigma_t,
          move_chance_t=move_chance_t,
          move_chance_s=move_chance_s,
          cache=cache)
      else:
        # Guided generation
        if self.config.guidance.method == 'cbg':
          xs, q_xs, cache = self._cbg_denoise(
            classifier_model=classifier_model,
            conditioning_class=self.config.guidance.condition,
            gamma=self.config.guidance.gamma,
            use_approx=self.config.guidance.use_approx,
            xt=xt,
            time_conditioning=sigma_t,
            move_chance_t=move_chance_t,
            move_chance_s=move_chance_s,
            cache=cache)
        elif self.config.guidance.method == 'cbg_antibiotic':
          xs, q_xs, cache = self._cbg_denoise_antibiotic(
            classifier_model=classifier_model,
            conditioning_class=self.config.guidance.condition,
            gamma=self.config.guidance.gamma,
            use_approx=self.config.guidance.use_approx,
            xt=xt,
            time_conditioning=sigma_t,
            move_chance_t=move_chance_t,
            move_chance_s=move_chance_s,
            cache=cache,
            step=i+1)
        elif self.config.guidance.method == 'cbg_antibiotic_remdm_loop':
          xs, q_xs, cache = self._cbg_denoise_antibiotic_remdm_loop(
            classifier_model=classifier_model,
            conditioning_class=self.config.guidance.condition,
            gamma=self.config.guidance.gamma,
            use_approx=self.config.guidance.use_approx,
            xt=xt,
            time_conditioning=sigma_t,
            move_chance_t=move_chance_t,
            move_chance_s=move_chance_s,
            cache=cache,
            step=i+1,
            t=t,
            dt=dt)
        else:
          raise NotImplementedError(
            f"Guidance method {self.config.guidance.method} not implemented.")

      # Update progress bar
      if hasattr(pbar, 'set_postfix'):
        pbar.set_postfix(
          NFEs=NFEs,
          prob_check=(q_xs.sum() / xt.numel()).item(),
          nan_check=bool(q_xs.isnan().sum() > 0))

      # Disable caching if samples changed or time conditioning is on
      if (not getattr(self.config.sampling, 'use_cache', False) or
          not torch.allclose(xs, xt)):
        cache = None

      xt = xs

    # Final denoising pass
    min_t = timesteps[-1].item()
    t = min_t * torch.ones(xt.shape[0], 1, device=self.device)
    unet_conditioning = self.noise(t)[0]
    xt = self.forward(xt, unet_conditioning).argmax(dim=-1)

    return xt

  def _ddpm_denoise(
    self,
    xt: torch.tensor,
    time_conditioning: torch.tensor,
    move_chance_t: torch.tensor,
    move_chance_s: torch.tensor,
    cache: typing.Optional[typing.Dict[str, torch.Tensor]] = None,
  ) -> typing.Tuple[torch.tensor, torch.tensor, typing.Dict[str, torch.tensor]]:
    """Standard DDPM denoising step without guidance.

    Args:
      xt: Current noisy sample of shape (B, L)
      time_conditioning: Noise level sigma_t of shape (B,)
      move_chance_t: Corruption probability at time t of shape (B, 1, 1)
      move_chance_s: Corruption probability at time s of shape (B, 1, 1)
      cache: Optional cached forward pass results

    Returns:
      Tuple of (sampled next state, posterior probabilities, cache dict)
    """
    # Compute x_theta from diffusion model
    if cache is not None:
      log_x_theta = cache['log_x_theta']
    else:
      log_x_theta = self.forward(xt, time_conditioning)
      if getattr(self.config.sampling, 'use_float64', False):
        log_x_theta = log_x_theta.to(torch.float64)

    x_theta = log_x_theta.exp()

    # Compute posterior distribution
    if self.diffusion == 'absorbing_state':
      # For absorbing state diffusion (masked language model)
      q_xs = x_theta * (move_chance_t - move_chance_s)
      q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
      q_xs /= move_chance_t
    elif self.diffusion == 'uniform':
      # For uniform diffusion
      q_xs = self._compute_posterior(
        x=x_theta,
        xt=xt,
        alpha_s=1 - move_chance_s,
        alpha_t=1 - move_chance_t)
    else:
      raise NotImplementedError(
        f"Diffusion type {self.diffusion} not implemented.")

    # Sample from posterior
    xs = _sample_categorical(q_xs)

    # For absorbing state, keep non-masked tokens fixed
    if self.diffusion == 'absorbing_state':
      copy_flag = (xt != self.mask_index).to(torch.bool)
      q_xs[copy_flag] = 0.0
      q_xs[copy_flag, xt[copy_flag]] = 1.0
      xs = torch.where(copy_flag, xt, xs)

    return xs, q_xs, {'log_x_theta': log_x_theta}

  def _cbg_denoise(
      self,
      conditioning_class: int,
      gamma: float,
      classifier_model: classifier.Classifier,
      xt: torch.tensor,
      time_conditioning: torch.tensor,
      move_chance_t: torch.tensor,
      move_chance_s: torch.tensor,
      use_approx: bool = False,
      cache: typing.Optional[typing.Dict[str, torch.Tensor]] = None,
  ) -> typing.Tuple[torch.tensor, torch.tensor, typing.Dict[str, torch.tensor]]:
    """Classifier-based guidance (CBG) denoising step.

    Combines diffusion model predictions with classifier gradients to guide
    generation toward desired properties.

    Args:
      conditioning_class: Target class for guidance
      gamma: Guidance strength parameter
      classifier_model: Trained classifier for guidance
      xt: Current noisy sample
      time_conditioning: Current timestep
      move_chance_t: Corruption probability at time t
      move_chance_s: Corruption probability at time s (t - dt)
      use_approx: Whether to use first-order gradient approximation
      cache: Optional cached values for efficiency

    Returns:
      Tuple of (denoised sample, guided probabilities, cache dict)
    """
    if cache is not None:
      log_x_theta = cache['log_x_theta']
      classifier_log_prob = cache['classifier_log_prob']
    else:
      # Diffusion model prediction
      log_x_theta = self.forward(xt, time_conditioning)

      # Classifier guidance
      if use_approx:
        # First-order gradient approximation
        xt_one_hot = torch.nn.functional.one_hot(xt, self.vocab_size).to(torch.float)
        with torch.enable_grad():
          xt_one_hot.requires_grad_(True)
          classifier_log_prob_xt = classifier_model.get_log_probs(xt_one_hot, time_conditioning)
          classifier_log_prob_xt[..., conditioning_class].sum().backward()
          grad_log_prob_xt = xt_one_hot.grad

        classifier_log_prob_ratio = (
            grad_log_prob_xt - (xt_one_hot * grad_log_prob_xt).sum(dim=-1, keepdim=True)
        ).detach().requires_grad_(False)

        classifier_log_prob = (
            classifier_log_prob_ratio + classifier_log_prob_xt[..., conditioning_class][..., None, None]
        ).detach().requires_grad_(False)
      else:
        # Exact computation via all possible transitions
        # Reference: https://github.com/hnisonoff/discrete_guidance/blob/main/src/fm_utils.py#L441
        bsz, seq_len = xt.shape
        # Create bsz*seq_len*N copies of input sequences
        xt_expand = xt.unsqueeze(1).repeat(1, seq_len * self.vocab_size, 1)
        xt_expand = xt_expand.view(-1, seq_len)

        # Create indices for all possible transitions
        jump_idx = torch.arange(seq_len * self.vocab_size).to(xt.device)
        jump_idx = jump_idx.repeat(bsz, 1).flatten()

        # Create tensor for states after one transition
        xt_jumps = xt_expand.clone()

        # Calculate transition dimensions and new values
        jump_dims = jump_idx // self.vocab_size
        jump_states = jump_idx % self.vocab_size

        # Apply transitions
        xt_jumps[
          torch.arange(jump_idx.size(0), device=xt.device),
          jump_dims,
        ] = jump_states

        classifier_log_prob = classifier_model.get_log_probs(
          xt_jumps, time_conditioning.repeat(seq_len * self.vocab_size)
        )[..., conditioning_class].reshape(bsz, seq_len, self.vocab_size)

    # Compute unguided posterior
    if self.diffusion == 'absorbing_state':
      diffusion_log_probs = log_x_theta + torch.log(1. - (move_chance_s / move_chance_t))
      diffusion_log_probs[..., self.mask_index] = torch.log(move_chance_s / move_chance_t)[:, :, 0]
      diffusion_log_probs.detach()
    elif self.diffusion == 'uniform':
      diffusion_log_probs = self._compute_posterior(
        x=log_x_theta.exp(),
        xt=xt,
        alpha_s=1 - move_chance_s,
        alpha_t=1 - move_chance_t).log()
    else:
      raise NotImplementedError(f"Diffusion type {self.diffusion} not implemented.")

    # Apply guidance
    with torch.no_grad():
      if self.diffusion == 'absorbing_state':
        guided_log_probs = (gamma * classifier_log_prob) + diffusion_log_probs
        copy_flag = (xt != self.mask_index)
        guided_log_probs[copy_flag] = self.neg_infinity
        guided_log_probs[copy_flag, xt[copy_flag]] = 0.0
      elif self.diffusion == 'uniform':
        guided_log_probs = (gamma * classifier_log_prob) + diffusion_log_probs
      else:
        raise NotImplementedError(f"Diffusion type {self.diffusion} not implemented.")

    guided_probs = guided_log_probs.softmax(dim=-1)
    # Sample from guided posterior
    xs = _sample_categorical(guided_probs)
    if self.diffusion == 'absorbing_state':
      xs = torch.where(copy_flag.to(bool), xt, xs)
    return xs, guided_probs, {'log_x_theta': log_x_theta,
                              'classifier_log_prob': classifier_log_prob}

  def _cbg_denoise_antibiotic(
      self,
      conditioning_class: int,
      gamma: float,
      classifier_model: classifier.Classifier,
      xt: torch.tensor,
      time_conditioning: torch.tensor,
      move_chance_t: torch.tensor,
      move_chance_s: torch.tensor,
      use_approx: bool = False,
      cache: typing.Optional[typing.Dict[str, torch.Tensor]] = None,
      step: int = None,
  ) -> typing.Tuple[torch.tensor, torch.tensor, typing.Dict[str, torch.tensor]]:
    """Antibiotic-specific guided generation with MIC targeting.

    Extended CBG method for antibiotic generation with:
    - Nucleus sampling for diversity
    - Variable-length sequence support
    - Forbidden token filtering (., I, Sn, Br)
    - Debug printing for monitoring

    Args:
      conditioning_class: Not used (kept for API compatibility)
      gamma: Guidance strength
      classifier_model: Regressor predicting MIC values
      xt: Current noisy sample
      time_conditioning: Current timestep
      move_chance_t: Corruption probability at time t
      move_chance_s: Corruption probability at time s
      use_approx: Whether to use gradient approximation (recommended: True)
      cache: Optional cached values
      step: Current sampling step

    Returns:
      Tuple of (denoised sample, guided probabilities, cache dict)
    """
    if cache is not None:
      log_x_theta = cache['log_x_theta']
      classifier_log_prob = cache['classifier_log_prob']
    else:
      # Diffusion model prediction
      log_x_theta = self.forward(xt, time_conditioning)

      # Apply nucleus sampling for diversity
      if self.config.sampling.nucleus_p < 1:
        p_x0 = log_x_theta.exp()
        sorted_probs, sorted_indices = torch.sort(p_x0, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        top_p_mask = cumulative_probs <= self.config.sampling.nucleus_p
        top_p_mask[..., 0] = True
        nucleus_probs = sorted_probs * top_p_mask
        nucleus_probs /= nucleus_probs.sum(dim=-1, keepdim=True)
        p_x0 = torch.zeros_like(p_x0).scatter_(-1, sorted_indices, nucleus_probs)
        log_x_theta = torch.log(p_x0 + 1e-8)

      # Classifier guidance (MIC regressor)
      if use_approx:
        xt_one_hot = torch.nn.functional.one_hot(xt, self.vocab_size).to(torch.float)

        # Handle variable-length sequences
        if self.config.guidance.var_length:
          backup_shape = xt_one_hot.shape
          xt_one_hot_backup = xt_one_hot
          xt_one_hot = xt_one_hot[:, :self.config.sampling.target_length, :]

        with torch.enable_grad():
          xt_one_hot.requires_grad_(True)
          classifier_log_prob_xt = classifier_model.get_log_probs_antibiotic_guaidance(
            xt_one_hot, time_conditioning, step=step)
          classifier_log_prob_xt.sum().backward()
          grad_log_prob_xt = xt_one_hot.grad

          if self.config.guidance.var_length:
            true_grad = torch.zeros(backup_shape, device=xt_one_hot.device)
            true_grad[:, :self.config.sampling.target_length, :] = grad_log_prob_xt
            grad_log_prob_xt = true_grad
            xt_one_hot = xt_one_hot_backup

        classifier_log_prob_ratio = (
            grad_log_prob_xt - (xt_one_hot * grad_log_prob_xt).sum(dim=-1, keepdim=True)
        ).detach().requires_grad_(False)

        classifier_log_prob = (
            classifier_log_prob_ratio + classifier_log_prob_xt[..., None]
        ).detach().requires_grad_(False)

        # Debug printing
        print(f' classifier_log_prob_xt min: {classifier_log_prob_xt.min()}')
        print(f' classifier_log_prob_xt max: {classifier_log_prob_xt.max()}')
        print(f' classifier_log_prob_xt mean: {classifier_log_prob_xt.mean()}')
        print(f' classifier_log_prob_ratio min: {classifier_log_prob_ratio.min()}')
        print(f' classifier_log_prob_ratio max: {classifier_log_prob_ratio.max()}')
        print(f' classifier_log_prob_ratio mean: {classifier_log_prob_ratio.mean()}')
        print(f' classifier_log_prob min: {classifier_log_prob.min()}')
        print(f' classifier_log_prob max: {classifier_log_prob.max()}')
        print(f' classifier_log_prob mean: {classifier_log_prob.mean()}')
      else:
        # Exact computation (memory-intensive)
        bsz, seq_len = xt.shape
        xt_expand = xt.unsqueeze(1).repeat(1, seq_len * self.vocab_size, 1)
        xt_expand = xt_expand.view(-1, seq_len)

        jump_idx = torch.arange(seq_len * self.vocab_size).to(xt.device)
        jump_idx = jump_idx.repeat(bsz, 1).flatten()
        xt_jumps = xt_expand.clone()
        jump_dims = jump_idx // self.vocab_size
        jump_states = jump_idx % self.vocab_size

        xt_jumps[
          torch.arange(jump_idx.size(0), device=xt.device),
          jump_dims,
        ] = jump_states

        classifier_log_prob = classifier_model.get_log_probs_antibiotic_guaidance(
          xt_jumps, time_conditioning.repeat(seq_len * self.vocab_size)
        ).reshape(bsz, seq_len, self.vocab_size)

    # Compute unguided posterior
    if self.diffusion == 'absorbing_state':
      diffusion_log_probs = log_x_theta + torch.log(1. - (move_chance_s / move_chance_t))
      diffusion_log_probs[..., self.mask_index] = torch.log(move_chance_s / move_chance_t)[:, :, 0]
      diffusion_log_probs.detach()

      print(f' diffusion_log_probs min: {diffusion_log_probs.min()}')
      print(f' diffusion_log_probs max: {diffusion_log_probs.max()}')
      print(f' diffusion_log_probs mean: {diffusion_log_probs.mean()}')

    elif self.diffusion == 'uniform':
      diffusion_log_probs = self._compute_posterior(
        x=log_x_theta.exp(),
        xt=xt,
        alpha_s=1 - move_chance_s,
        alpha_t=1 - move_chance_t).log()
    else:
      raise NotImplementedError(f"Diffusion type {self.diffusion} not implemented.")

    # Apply guidance
    with torch.no_grad():
      if self.diffusion == 'absorbing_state':
        guided_log_probs = (gamma * classifier_log_prob) + diffusion_log_probs
        copy_flag = (xt != self.mask_index)
        guided_log_probs[copy_flag] = self.neg_infinity
        guided_log_probs[copy_flag, xt[copy_flag]] = 0.0
      elif self.diffusion == 'uniform':
        guided_log_probs = (gamma * classifier_log_prob) + diffusion_log_probs
      else:
        raise NotImplementedError(f"Diffusion type {self.diffusion} not implemented.")

    # Forbid specific tokens (chemical elements that shouldn't be in small molecules)
    vocab = self.tokenizer.get_vocab()
    token_id_forbid = [vocab[word] for word in vocab.keys() if '.' in word]
    token_id_forbid += [vocab[word] for word in vocab.keys() if 'I' in word]
    token_id_forbid += [vocab[word] for word in vocab.keys() if 'Sn' in word]
    token_id_forbid += [vocab[word] for word in vocab.keys() if 'Br' in word]
    guided_log_probs[:, :self.config.sampling.target_length, token_id_forbid] = self.neg_infinity

    print(f'\n guided_log_probs min: {guided_log_probs.min()}')
    print(f' guided_log_probs max: {guided_log_probs.max()}')
    print(f' guided_log_probs mean: {guided_log_probs.mean()}')

    guided_probs = guided_log_probs.softmax(dim=-1)

    print(f'\n guided_probs min: {guided_probs.min()}')
    print(f' guided_probs max: {guided_probs.max()}')
    print(f' guided_probs mean: {guided_probs.mean()}')

    # Sample from guided posterior
    xs = _sample_categorical(guided_probs)
    if self.diffusion == 'absorbing_state':
      xs = torch.where(copy_flag.to(bool), xt, xs)
    return xs, guided_probs, {'log_x_theta': log_x_theta,
                              'classifier_log_prob': classifier_log_prob}

  def _cbg_denoise_antibiotic_remdm_loop(
      self,
      conditioning_class: int,
      gamma: float,
      classifier_model: classifier.Classifier,
      xt: torch.tensor,
      time_conditioning: torch.tensor,
      move_chance_t: torch.tensor,
      move_chance_s: torch.tensor,
      use_approx: bool = False,
      cache: typing.Optional[typing.Dict[str, torch.Tensor]] = None,
      step: int = None,
      t=None,
      dt=None,
  ) -> typing.Tuple[torch.tensor, torch.tensor, typing.Dict[str, torch.tensor]]:
    """Hybrid MDLM/ReMDM guided generation for antibiotics.

    Advanced guided generation combining:
    - MDLM (Masked Diffusion Language Model) for early/late steps
    - ReMDM (Reparameterized Masked Diffusion Model) for middle steps
    - Time-adaptive guidance strength (gamma)
    - Variable-length sequence support
    - Nucleus sampling
    - Token filtering

    The sampling process is divided into three phases:
    1. Early phase (t > t_on): Use MDLM with strong guidance (gamma_l)
    2. Middle phase (t_off < t <= t_on): Use ReMDM with weaker guidance (gamma_s)
    3. Late phase (t <= t_off): Use MDLM with strong guidance (gamma_l)

    Args:
      conditioning_class: Not used (kept for API compatibility)
      gamma: Initial guidance strength (overridden by time-adaptive values)
      classifier_model: Regressor for MIC prediction
      xt: Current noisy sample
      time_conditioning: Current timestep
      move_chance_t: Corruption probability at time t (may be recomputed)
      move_chance_s: Corruption probability at time s (may be recomputed)
      use_approx: Whether to use gradient approximation
      cache: Optional cached values
      step: Current sampling step
      t: Current time as scalar tensor
      dt: Time step size

    Returns:
      Tuple of (denoised sample, guided probabilities, cache dict)
    """
    if cache is not None:
      log_x_theta = cache['log_x_theta']
      classifier_log_prob = cache['classifier_log_prob']
    else:
      # Diffusion model prediction
      log_x_theta = self.forward(xt, time_conditioning)

      # Apply nucleus sampling
      if self.config.sampling.nucleus_p < 1:
        p_x0 = log_x_theta.exp()
        sorted_probs, sorted_indices = torch.sort(p_x0, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        top_p_mask = cumulative_probs <= self.config.sampling.nucleus_p
        top_p_mask[..., 0] = True
        nucleus_probs = sorted_probs * top_p_mask
        nucleus_probs /= nucleus_probs.sum(dim=-1, keepdim=True)
        log_x_theta = torch.zeros_like(p_x0).scatter_(-1, sorted_indices, nucleus_probs)
        # Note: log_x_theta is now probability (not log) due to nucleus sampling

      # Classifier guidance
      if use_approx:
        xt_one_hot = torch.nn.functional.one_hot(xt, self.vocab_size).to(torch.float)

        if self.config.guidance.var_length:
          backup_shape = xt_one_hot.shape
          xt_one_hot_backup = xt_one_hot
          xt_one_hot = xt_one_hot[:, :self.config.sampling.target_length, :]

        with torch.enable_grad():
          xt_one_hot.requires_grad_(True)
          classifier_log_prob_xt = classifier_model.get_log_probs_antibiotic_guaidance(
            xt_one_hot, time_conditioning, step=step)
          classifier_log_prob_xt.sum().backward()
          grad_log_prob_xt = xt_one_hot.grad

          if self.config.guidance.var_length:
            true_grad = torch.zeros(backup_shape, device=xt_one_hot.device)
            true_grad[:, :self.config.sampling.target_length, :] = grad_log_prob_xt
            grad_log_prob_xt = true_grad
            xt_one_hot = xt_one_hot_backup

        classifier_log_prob_ratio = (
            grad_log_prob_xt - (xt_one_hot * grad_log_prob_xt).sum(dim=-1, keepdim=True)
        ).detach().requires_grad_(False)

        classifier_log_prob = (
            classifier_log_prob_ratio + classifier_log_prob_xt[..., None]
        ).detach().requires_grad_(False)

        # Debug printing
        print(f' classifier_log_prob_xt min: {classifier_log_prob_xt.min()}')
        print(f' classifier_log_prob_xt max: {classifier_log_prob_xt.max()}')
        print(f' classifier_log_prob_xt mean: {classifier_log_prob_xt.mean()}')
        print(f' classifier_log_prob_ratio min: {classifier_log_prob_ratio.min()}')
        print(f' classifier_log_prob_ratio max: {classifier_log_prob_ratio.max()}')
        print(f' classifier_log_prob_ratio mean: {classifier_log_prob_ratio.mean()}')
        print(f' classifier_log_prob min: {classifier_log_prob.min()}')
        print(f' classifier_log_prob max: {classifier_log_prob.max()}')
        print(f' classifier_log_prob mean: {classifier_log_prob.mean()}')
      else:
        # Exact computation
        bsz, seq_len = xt.shape
        xt_expand = xt.unsqueeze(1).repeat(1, seq_len * self.vocab_size, 1)
        xt_expand = xt_expand.view(-1, seq_len)

        jump_idx = torch.arange(seq_len * self.vocab_size).to(xt.device)
        jump_idx = jump_idx.repeat(bsz, 1).flatten()
        xt_jumps = xt_expand.clone()
        jump_dims = jump_idx // self.vocab_size
        jump_states = jump_idx % self.vocab_size

        xt_jumps[
          torch.arange(jump_idx.size(0), device=xt.device),
          jump_dims,
        ] = jump_states

        classifier_log_prob = classifier_model.get_log_probs_antibiotic_guaidance(
          xt_jumps, time_conditioning.repeat(seq_len * self.vocab_size)
        ).reshape(bsz, seq_len, self.vocab_size)

    # Compute time-adaptive posterior (MDLM vs ReMDM)
    if self.diffusion == 'absorbing_state':
      # Handle nucleus sampling output
      if self.config.sampling.nucleus_p < 1:
        p_x0 = log_x_theta  # Already probability from nucleus sampling
      else:
        p_x0 = log_x_theta.exp()

      if t.ndim > 1:
        t = t.squeeze(-1)
      time = t[0].item()

      # Time-adaptive guidance strategy
      if time > self.config.sampling.remdm.t_on:
        # Early phase: MDLM with strong guidance
        gamma = self.config.guidance.var_gamma.gamma_l
        self.config.guidance.reg_guide_weight = 1.0
        self.config.guidance.cls_guide_weight = 0.0
        move_chance_t = (1 - (1 - t) * self.config.sampling.remdm.alpha_on / (
          1 - self.config.sampling.remdm.t_on))[:, None, None]
        move_chance_s = (1 - (1 - t + dt) * self.config.sampling.remdm.alpha_on / (
          1 - self.config.sampling.remdm.t_on))[:, None, None]
      elif time <= self.config.sampling.remdm.t_off:
        # Late phase: MDLM with strong guidance
        gamma = self.config.guidance.var_gamma.gamma_l
        self.config.guidance.reg_guide_weight = 1.0
        self.config.guidance.cls_guide_weight = 0.0
        move_chance_t = (t * (1 - self.config.sampling.remdm.alpha_on) /
                        self.config.sampling.remdm.t_off)[:, None, None]
        move_chance_s = ((t - dt) * (1 - self.config.sampling.remdm.alpha_on) /
                        self.config.sampling.remdm.t_off)[:, None, None]
      else:
        # Middle phase: ReMDM with weaker guidance
        gamma = self.config.guidance.var_gamma.gamma_s
        self.config.guidance.reg_guide_weight = 0.0
        self.config.guidance.cls_guide_weight = 1.0
        move_chance_t, move_chance_s = None, None

      # Compute posterior based on phase
      if time > self.config.sampling.remdm.t_on or time <= self.config.sampling.remdm.t_off:
        # MDLM posterior
        q_xs = p_x0 * (1. - (move_chance_s / move_chance_t))
        q_xs[:, :, self.mask_index] = (move_chance_s / move_chance_t)[:, :, 0]
        diffusion_log_probs = torch.log(q_xs + 1e-8)
      else:
        # ReMDM posterior
        sigma = self.config.sampling.remdm.eta
        q_xs = p_x0 * (1 - sigma)
        q_xs[..., self.mask_index] = sigma
        q_xs_2 = p_x0 * ((self.config.sampling.remdm.alpha_on -
                         (1 - sigma) * self.config.sampling.remdm.alpha_on) /
                        (1 - self.config.sampling.remdm.alpha_on))
        q_xs_2[..., self.mask_index] = (1 - self.config.sampling.remdm.alpha_on -
                                       self.config.sampling.remdm.alpha_on * sigma) / (
                                       1 - self.config.sampling.remdm.alpha_on)
        copy_flag = (xt != self.mask_index).to(torch.bool)
        q_xs = torch.where(copy_flag.unsqueeze(-1), q_xs, q_xs_2)
        diffusion_log_probs = torch.log(q_xs + 1e-8)

      print(f' diffusion_log_probs min: {diffusion_log_probs.min()}')
      print(f' diffusion_log_probs max: {diffusion_log_probs.max()}')
      print(f' diffusion_log_probs mean: {diffusion_log_probs.mean()}')

    elif self.diffusion == 'uniform':
      diffusion_log_probs = self._compute_posterior(
        x=log_x_theta.exp(),
        xt=xt,
        alpha_s=1 - move_chance_s,
        alpha_t=1 - move_chance_t).log()
    else:
      raise NotImplementedError(f"Diffusion type {self.diffusion} not implemented.")

    # Apply guidance
    with torch.no_grad():
      if self.diffusion == 'absorbing_state':
        # Phase-specific guidance application
        if time > self.config.sampling.remdm.t_on or time <= self.config.sampling.remdm.t_off:
          # MDLM: only update masked positions
          guided_log_probs = (gamma * classifier_log_prob) + diffusion_log_probs
          copy_flag = (xt != self.mask_index)
          guided_log_probs[copy_flag] = self.neg_infinity
          guided_log_probs[copy_flag, xt[copy_flag]] = 0.0
        else:
          # ReMDM: allow remasking
          guided_log_probs = (gamma * classifier_log_prob) + diffusion_log_probs
          copy_flag = (xt != self.mask_index)
          guided_log_probs_copy = guided_log_probs.clone()
          guided_log_probs[copy_flag] = self.neg_infinity
          guided_log_probs[copy_flag, xt[copy_flag]] = guided_log_probs_copy[copy_flag, xt[copy_flag]]
          guided_log_probs[copy_flag, self.mask_index * torch.ones_like(xt[copy_flag])] = \
            guided_log_probs_copy[copy_flag, self.mask_index * torch.ones_like(xt[copy_flag])]

          # Prevent remasking padding tokens
          guided_log_probs[:, self.config.sampling.target_length:, :] = self.neg_infinity
          guided_log_probs[:, self.config.sampling.target_length:, self.tokenizer.pad_token_id] = 0.0

        # Forbid specific tokens
        vocab = self.tokenizer.get_vocab()
        token_id_forbid = []
        # Full periodic table forbidden list available in comments if needed:
        # forbidden_list = ['.', 'He', 'Li', 'Be', 'B', 'Ne', ...]
        forbidden_list = ['.', 'I', 'Sn', 'Br']
        for forbiden_id in forbidden_list:
          token_id_forbid += [vocab[word] for word in vocab.keys() if forbiden_id in word]
        guided_log_probs[:, :self.config.sampling.target_length, token_id_forbid] = self.neg_infinity

      elif self.diffusion == 'uniform':
        guided_log_probs = (gamma * classifier_log_prob) + diffusion_log_probs
      else:
        raise NotImplementedError(f"Diffusion type {self.diffusion} not implemented.")

    print(f'\n guided_log_probs min: {guided_log_probs.min()}')
    print(f' guided_log_probs max: {guided_log_probs.max()}')
    print(f' guided_log_probs mean: {guided_log_probs.mean()}')

    guided_probs = guided_log_probs.softmax(dim=-1)

    print(f'\n guided_probs min: {guided_probs.min()}')
    print(f' guided_probs max: {guided_probs.max()}')
    print(f' guided_probs mean: {guided_probs.mean()}')

    # Sample from guided posterior
    xs = _sample_categorical(guided_probs)
    xs[:, 0] = self.tokenizer.cls_token_id  # Ensure CLS token at start

    if self.diffusion == 'absorbing_state':
      if time > self.config.sampling.remdm.t_on or time <= self.config.sampling.remdm.t_off:
        xs = torch.where(copy_flag.to(bool), xt, xs)

    return xs, guided_probs, {'log_x_theta': log_x_theta,
                              'classifier_log_prob': classifier_log_prob}

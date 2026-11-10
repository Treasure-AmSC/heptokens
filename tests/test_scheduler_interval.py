"""ScheduledOptimiserMixin: opt-in per-step LR scheduling."""

from functools import partial

import pytest
import torch

from heptokens.models.coders import Decoder, Encoder
from heptokens.models.utils import linear_warmup_cosine_decay
from heptokens.models.vq_vae import LitVqVae


def _model(**kwargs) -> LitVqVae:
    return LitVqVae(
        encoder=partial(Encoder),
        decoder=partial(Decoder),
        codebook_size=16,
        codebook_dim=4,
        num_quantizers=2,
        data_sample=torch.zeros(3, 5),
        optimizer=partial(torch.optim.Adam, lr=3e-4),
        scheduler=partial(
            linear_warmup_cosine_decay,
            warmup_steps=10,
            total_steps=100,
            final_factor=1e-6 / 3e-4,
            init_factor=1e-2,
        ),
        **kwargs,
    )


def test_scheduler_interval_defaults_to_epoch():
    cfg = _model().configure_optimizers()
    assert cfg["lr_scheduler"]["interval"] == "epoch"


@pytest.mark.parametrize("interval", ["step", "epoch"])
def test_scheduler_interval_from_hparams(interval):
    model = _model(scheduler_interval=interval)
    assert model.hparams["scheduler_interval"] == interval
    assert model.configure_optimizers()["lr_scheduler"]["interval"] == interval


def test_linear_warmup_cosine_decay_shape():
    cfg = _model(scheduler_interval="step").configure_optimizers()
    opt, sched = cfg["optimizer"], cfg["lr_scheduler"]["scheduler"]
    lrs = []
    for _ in range(120):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] == pytest.approx(3e-6)  # init_factor * lr
    assert lrs[10] == pytest.approx(3e-4)  # peak at end of warmup
    assert lrs[100] == pytest.approx(1e-6)  # floor at total_steps
    assert all(a >= b for a, b in zip(lrs[10:100], lrs[11:101]))  # monotone decay

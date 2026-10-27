import torch
from torch import nn

from ptqkit.measure import perplexity


class UniformLM(nn.Module):
    def __init__(self, vocab: int) -> None:
        super().__init__()
        self.vocab = vocab

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.nn.Module:
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], self.vocab)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            loss = nn.functional.cross_entropy(
                shift_logits.reshape(-1, self.vocab),
                shift_labels.reshape(-1),
                ignore_index=-100,
            )
        return type("Out", (), {"loss": loss, "logits": logits})()


class DeltaLM(nn.Module):
    def __init__(self, vocab: int) -> None:
        super().__init__()
        self.vocab = vocab

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.nn.Module:
        logits = torch.full((input_ids.shape[0], input_ids.shape[1], self.vocab), -100.0)
        if input_ids.shape[1] > 1:
            logits[:, :-1, :].scatter_(2, input_ids[:, 1:].unsqueeze(-1), 50.0)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            loss = nn.functional.cross_entropy(
                shift_logits.reshape(-1, self.vocab),
                shift_labels.reshape(-1),
                ignore_index=-100,
            )
        return type("Out", (), {"loss": loss, "logits": logits})()


def test_uniform_perplexity_is_vocab() -> None:
    vocab = 10
    ids = torch.randint(0, vocab, (1, 20))
    stats = perplexity(UniformLM(vocab), ids, context=8, stride=4)
    assert abs(stats["perplexity"] - vocab) < 1e-4
    assert stats["tokens"] > 0


def test_overlapping_windows_score_every_token_after_the_first() -> None:
    vocab = 10
    ids = torch.arange(20, dtype=torch.long).view(1, 20) % vocab
    overlap = perplexity(UniformLM(vocab), ids, context=8, stride=4)
    disjoint = perplexity(UniformLM(vocab), ids, context=8, stride=8)
    assert overlap["tokens"] == 19
    assert disjoint["tokens"] == 17
    assert abs(overlap["perplexity"] - vocab) < 1e-4


def test_delta_perplexity_is_near_one() -> None:
    vocab = 12
    ids = torch.randint(0, vocab, (1, 24))
    for stride in (8, 4):
        stats = perplexity(DeltaLM(vocab), ids, context=8, stride=stride)
        assert stats["perplexity"] < 1.01

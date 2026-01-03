from dataclasses import dataclass
from transformers.trainer_utils import EvalPrediction
from typing import Dict
from typing import TYPE_CHECKING, Optional
import numpy as np
from ..data.constants import IGNORE_INDEX
from ..extras.misc import numpify

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


@dataclass
class TranslateMetric:
    r"""Compute text  bleu and chrF and support `batch_eval_metrics`.

    Wraps the tokenizer into metric functions, used in CustomSeq2SeqTrainer.
    """

    tokenizer: "PreTrainedTokenizer"

    def __post_init__(self):
        from evaluate import load
        from sacrebleu import corpus_bleu

        self.corpus_bleu = corpus_bleu
        self.charf = load("chrf")

    def __call__(
        self, eval_preds: "EvalPrediction", compute_result: bool = True
    ) -> Optional[dict[str, float]]:
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)

        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        bleu_score = self.corpus_bleu(
            decoded_preds, [[label] for label in decoded_labels]
        )
        charf_score = self.charf.compute(
            predictions=decoded_preds, references=decoded_labels
        )

        return {
            "bleu": round(bleu_score.score, 6),
            "chrf": round(charf_score["score"], 6),
        }

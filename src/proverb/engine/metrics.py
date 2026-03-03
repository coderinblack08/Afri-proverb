from dataclasses import dataclass
from transformers.trainer_utils import EvalPrediction
from typing import Dict, List
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
        self.comet = load("comet")

    def __call__(
        self, eval_preds: "EvalPrediction", compute_result: bool = True
    ) -> Optional[dict[str, float]]:
        if isinstance(eval_preds.label_ids, tuple):
            preds, labels, sources = (
                numpify(eval_preds.predictions),
                numpify(eval_preds.label_ids[0]),
                numpify(eval_preds.label_ids[1]),
            )
        else:
            preds, labels = (
                numpify(eval_preds.predictions),
                numpify(eval_preds.label_ids),
            )
            sources = None

        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)
        if sources is not None:
            sources = np.where(
                sources != IGNORE_INDEX, sources, self.tokenizer.pad_token_id
            )

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_sources = None
        if sources is not None:
            decoded_sources = self.tokenizer.batch_decode(
                sources, skip_special_tokens=True
            )

        bleu_score = self.corpus_bleu(
            decoded_preds, [[label] for label in decoded_labels]
        )
        charf_score = self.charf.compute(
            predictions=decoded_preds,
            references=decoded_labels,
        )
        chrf_pp_score = self.charf.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            char_order=6,
            word_order=2,
        )
        ret = {
            "bleu": round(bleu_score.score, 6),
            "chrf": round(charf_score["score"], 6),
            "chrf++": round(chrf_pp_score["score"], 6),
        }

        if decoded_sources is not None:
            comet_score = self.comet.compute(
                predictions=decoded_preds,
                references=decoded_labels,
                sources=decoded_sources,
            )

            ret["comet"] = round(comet_score["mean_score"], 6)

        return ret


@dataclass
class TextTranslateMetric:
    include_comet: bool = True

    def __post_init__(self):
        from evaluate import load
        from sacrebleu import corpus_bleu

        self.corpus_bleu = corpus_bleu
        self.charf = load("chrf")
        self.comet = None
        if self.include_comet:
            try:
                self.comet = load("comet")
            except Exception:
                self.comet = None

    def __call__(
        self,
        predictions: List[str],
        references: List[str],
        sources: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        bleu_score = self.corpus_bleu(predictions, [[label] for label in references])
        charf_score = self.charf.compute(
            predictions=predictions,
            references=references,
        )
        chrf_pp_score = self.charf.compute(
            predictions=predictions,
            references=references,
            char_order=6,
            word_order=2,
        )
        ret = {
            "bleu": round(bleu_score.score, 6),
            "chrf": round(charf_score["score"], 6),
            "chrf++": round(chrf_pp_score["score"], 6),
        }

        if self.comet is not None and sources is not None and len(sources) == len(
            predictions
        ):
            comet_score = self.comet.compute(
                predictions=predictions,
                references=references,
                sources=sources,
            )
            ret["comet"] = round(comet_score["mean_score"], 6)

        return ret

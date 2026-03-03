from dataclasses import dataclass
import re
import subprocess
import tempfile
import time
from transformers.trainer_utils import EvalPrediction
from typing import Dict, List
from typing import TYPE_CHECKING, Optional
import numpy as np
from ..data.constants import IGNORE_INDEX
from ..extras.logging import get_logger
from ..extras.misc import numpify

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

logger = get_logger("proverb.engine.metrics")


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
        self.comet_mode = "disabled"
        if self.include_comet:
            try:
                self.comet = load("comet")
                self.comet_mode = "native"
            except Exception as exc:
                # Keep COMET available through an isolated sidecar environment to avoid
                # pin conflicts with the main runtime (torch/transformers versions).
                self.comet_mode = "sidecar"
                logger.warning_rank0(
                    "Falling back to sidecar COMET scoring because evaluate/comet could not be loaded: %s",
                    exc,
                )

    @staticmethod
    def _compute_comet_sidecar(
        predictions: List[str], references: List[str], sources: List[str]
    ) -> float:
        with tempfile.TemporaryDirectory(prefix="comet-score-") as tmp_dir:
            src_path = f"{tmp_dir}/sources.txt"
            mt_path = f"{tmp_dir}/predictions.txt"
            ref_path = f"{tmp_dir}/references.txt"

            with open(src_path, "w", encoding="utf-8") as f:
                f.write("\n".join(s.replace("\n", " ").strip() for s in sources))
            with open(mt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(s.replace("\n", " ").strip() for s in predictions))
            with open(ref_path, "w", encoding="utf-8") as f:
                f.write("\n".join(s.replace("\n", " ").strip() for s in references))

            cmd = [
                "uvx",
                "--from",
                "unbabel-comet==2.2.7",
                "--with",
                "setuptools<81",
                "comet-score",
                "-s",
                src_path,
                "-t",
                mt_path,
                "-r",
                ref_path,
                "--quiet",
                "--only_system",
            ]
            logger.info_rank0(
                "Starting COMET sidecar scoring for %d segments (first run may download models).",
                len(predictions),
            )
            start_time = time.time()
            proc = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
            output = (proc.stdout or "").strip()
            # comet-score --only_system prints a single score value.
            match = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", output)
            if not match:
                raise RuntimeError(
                    f"Could not parse COMET score from sidecar output: {output!r}"
                )
            score = float(match[-1])
            elapsed = time.time() - start_time
            logger.info_rank0(
                "COMET sidecar scoring finished in %.2fs with score %.6f",
                elapsed,
                score,
            )
            return score

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

        if sources is not None and len(sources) == len(predictions):
            if self.comet_mode == "native" and self.comet is not None:
                comet_score = self.comet.compute(
                    predictions=predictions,
                    references=references,
                    sources=sources,
                )
                ret["comet"] = round(comet_score["mean_score"], 6)
            elif self.comet_mode == "sidecar":
                ret["comet"] = round(
                    self._compute_comet_sidecar(predictions, references, sources), 6
                )

        return ret

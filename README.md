# Content-Budget Invariance in Multilingual Dense Retrieval

Code and results for the paper *Content-Budget Invariance in Multilingual Dense Retrieval: Evaluating the Impact of Tokenizer Fertility on Text Chunking Boundaries* (Robert Bota, MSc Computer Science, University of Zimbabwe, 2026).

The study tests whether the retrieval-optimal chunk budget of a frozen multilingual dense retriever is the same across languages once the budget is measured in content rather than subword tokens. Fertility-Scaled Budgeting (FSB) scales an English budget by each language's FLORES-200 token premium. Experiments use TyDi QA (English, Indonesian, Finnish, Swahili, Telugu), BGE-M3 and Multilingual-E5-base, and exact FAISS search.

## Contents

| Path | What it is |
|---|---|
| `FSB_Experiment.ipynb` | The Colab notebook (T4 GPU): token premiums, data preparation, chunking policies, encoding, retrieval, metrics, analysis and figures |
| `notebook_src/fsb_experiment.py` | Source of the notebook; `python notebook_src/build_ipynb.py` rebuilds the `.ipynb` from it |
| `results/phase0.json` | FLORES-200 token premiums and run environment |
| `qwen3_premiums.py`, `results/qwen3_premiums.json` | Side measurement for Section 5.2: FLORES-200 premiums under the Qwen3-Embedding-0.6B tokenizer (for example Telugu 7.05, Swahili 1.93), with BGE-M3 recounted as a check (identical token counts to `phase0.json`). CPU only: `pip install transformers && python qwen3_premiums.py` |
| `results/runs_summary.csv` | One row per run (encoder, language, policy, budget, chunks, nDCG@10, time) |
| `results/<encoder>/results_<encoder>.json` | H1 to H3 test results (likelihood-ratio test, bootstrap optima, equivalence tests) |
| `results/<encoder>/means_<encoder>.csv` | Mean nDCG@10, MRR@10, Recall@5, chunk Hit@5, AR@128 and AR@256 per configuration |
| `results/<encoder>/rq3_<encoder>.csv` | FSB against baselines B1 to B4: paired sign-flip tests with Holm correction |
| `results/<encoder>/robustness_<encoder>.csv` | FSB-W and FSB-S against FSB |
| `results/<encoder>/fig_budget_curves_<encoder>.png` | Budget curves (Figures 1 and 2 of the paper) |

`<encoder>` is `bge-m3` or `me5-base`.

**Per-question scores.** The per-question scores and retrieved chunk lists for every run (635 files, 126 MB) are attached to the GitHub release as `fsb_experiment_runs_2026-11-20.zip` (SHA-256 `e4a8fbe7e1bbc4ca40c5ebea7f23fc23384e0e297412b77f09b82bcc1975868f`). Unzip it into the notebook's working folder to re-run the analysis without a GPU.

## Reproducing

1. Open `FSB_Experiment.ipynb` in Google Colab with a T4 GPU. The notebook mounts Google Drive and writes to `MyDrive/fsb_experiment`.
2. Run all cells. The run loop is resumable and skips finished runs. Set `RUN_EXPERIMENTS = False` to run only the analysis on existing results.
3. All data are public: TyDi QA from the Hugging Face dataset `google-research-datasets/tydiqa` and FLORES-200 dev sentences. Random seed: 2026.

Full runs took 5.3 GPU-hours for BGE-M3 (170 runs) and 1.6 GPU-hours for Multilingual-E5-base (135 runs) on one T4.

## Licence

Code: MIT. Results files: CC BY 4.0. TyDi QA and FLORES-200 keep their own licences.

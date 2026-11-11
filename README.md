# Hindi Text Restoration & Translation Pipeline

**Author:** Gautam Jain
**Institute:** SVNIT Surat

A three-stage NLP pipeline that takes raw, unpunctuated, shuffled Hindi text and turns it into readable English:

```
raw Hindi (no punctuation, shuffled)
   │
   ▼  Task 1: Punctuation restoration      (DistilBERT-multilingual, fine-tuned)
punctuated Hindi sentences
   │
   ▼  Task 2: Sentence ordering + topic grouping   (mBERT next-sentence model, fine-tuned)
ordered sentences → groups of 5
   │
   ▼  Task 3: Translation                  (IndicTrans2 Indic→En distilled 200M, pre-trained)
English text, one paragraph per group
```

All models are between 50M and 200M parameters and every stage runs in **under 1.5 GB of GPU memory**. No large language models (GPT, Gemini, Claude) are used anywhere in the pipeline.

---

## Trained models (Google Drive)

| Model | Task | Link |
|---|---|---|
| Model 1: Punctuation restoration | Task 1 | `https://drive.google.com/drive/folders/1NRKbZUsm7Vu-e7wVQG-BoRNt0i9_0Xd6?usp=drive_link` |
| Model 2: Sentence-order scorer | Task 2 | `https://drive.google.com/drive/folders/1NRKbZUsm7Vu-e7wVQG-BoRNt0i9_0Xd6?usp=drive_link` |
| Model 3: IndicTrans2 (used as released) | Task 3 | [ai4bharat/indictrans2-indic-en-dist-200M](https://huggingface.co/ai4bharat/indictrans2-indic-en-dist-200M) |

---

## GPU memory

Measured with `torch.cuda.max_memory_allocated()` on a Tesla T4 (Kaggle).

| Stage | Model size | Peak GPU memory |
|---|---|---|
| Task 1 training | 134.7M params (42.5M trainable) | **1.265 GB** |
| Task 1 inference | 134.7M | **0.938 GB** |
| Task 2 training | 177.9M params (28.9M trainable) | **0.899 GB** |
| Task 2 ordering | 177.9M | **< 1.45 GB** (hard cap enforced) |
| Task 3 translation | 211.8M | **0.812 GB** |

How memory was kept low:
- **Frozen embedding tables.** In multilingual BERT models the vocabulary table is about 92M parameters. Freezing it removes its gradients and optimizer state.
- **Partial layer freezing** (Task 2): only the top 4 of 12 transformer layers are trained.
- **Mixed precision (fp16)** and small per-step batches with gradient accumulation (effective batch size 16).
- **Adafactor optimizer** (Task 2), which keeps far less state than Adam.
- **Hard memory limit** (Tasks 2 and 3): `torch.cuda.set_per_process_memory_fraction` caps the process at 1.45 GB, so the limit cannot be exceeded.

---

## Data

- **Training data:** [AI4Bharat Sangraha](https://huggingface.co/datasets/ai4bharat/sangraha), `verified/hin` subset (cleaned, human-verified Hindi web text), streamed from Hugging Face.
  - Task 1: 8,000 documents, 80/20 document-level train/test split
  - Task 2: 6,000 documents, 80/10/10 train/validation/test split
- **Target data:** 7 raw samples (`sample-1.txt` … `sample-7.txt`), Hindi without punctuation and with sentences shuffled. `NULL` tokens in the raw files are removed before processing.
- **Translation benchmark:** FLORES-200 devtest (`hin_Deva` → `eng_Latn`).
- Random seed **42** everywhere, for reproducibility.

---

## Task 1: Punctuation restoration

**Approach:** token classification. Each word gets the label of the punctuation mark that follows it:

| Label | Meaning |
|---|---|
| `O` | no punctuation |
| `END` | sentence end (।) |
| `COMMA` | comma (,) |
| `QUESTION` | question mark (?) |

**Pipeline**
1. Punctuated Sangraha text is tokenised into words and punctuation marks. Decimal numbers such as `3.5` are kept as one word.
2. Each word is labelled with the punctuation after it. Training examples are 64-word windows.
3. Only the first sub-word of each word carries a label; the rest are ignored in the loss.
4. **Model:** `distilbert-base-multilingual-cased` with a 4-class token-classification head, trained for 3 epochs, learning rate 5e-5, fp16.
5. **Class-weighted loss** (square-root inverse frequency, capped at 5×), so the rarer marks (, and ?) are learned properly.
6. **Inference** on the samples uses overlapping 64-word windows (stride 48), so every word is predicted with context on both sides.

**Results** (test set, 670,779 words from held-out documents)

| Label | Precision | Recall | F1 |
|---|---|---|---|
| O | 0.989 | 0.965 | **0.977** |
| END (।) | 0.776 | 0.875 | **0.823** |
| COMMA (,) | 0.430 | 0.648 | **0.517** |
| QUESTION (?) | 0.452 | 0.562 | **0.501** |
| **Accuracy** | | | **0.950** |

---

## Task 2: Sentence ordering and grouping

**Approach:** learn which sentence follows which, then search for the best overall order.

**1. Pair scoring model.** `bert-base-multilingual-cased` (mBERT) comes with a pre-trained *Next Sentence Prediction* head. We fine-tune it on Hindi pairs built from Sangraha:
- positive: (sentence *i*, sentence *i+1*)
- negatives: the same pair reversed, and (sentence *i*, a random non-adjacent sentence from the same document)
- a special `<start>` marker teaches the model which sentence opens a text.

Fine-tuning raised pair-ranking quality (ROC-AUC) from **0.627** zero-shot to **0.706**.

**2. Ordering.** For a sample with *n* sentences, the model scores all *n × (n − 1)* ordered pairs. The scores are combined with TF-IDF similarity (weight α = 0.5, chosen on the validation set). **Beam search** (width 64) finds the highest-scoring order, and a **local refinement** step tries moving blocks of 1–3 sentences wherever that raises the total score.

**3. Topic grouping.** Some samples mix several topics (for example, sample-3 covers both Tijara and Hastinapur). Sentences are clustered with TF-IDF features and agglomerative clustering. The decision to split, and the threshold, are tuned automatically on synthetic mixtures of real documents. Each topic is ordered separately, then the topics are arranged by the model's transition scores.

**4. Grouping.** The final order is cut into groups of 5 sentences for translation.

**Results** (300 held-out test documents, 5–20 sentences each, shuffled)

| Method | Kendall τ | Correct adjacent pairs | Correct first sentence |
|---|---|---|---|
| Random order | -0.013 | 10.3% | 12.0% |
| TF-IDF similarity | 0.038 | 16.2% | 9.0% |
| mBERT (fine-tuned) | 0.168 | 26.5% | 36.3% |
| mBERT + TF-IDF | 0.192 | 26.9% | 36.3% |
| **Final: mBERT + TF-IDF + local refinement** (used for the samples) | **0.190** | **27.4%** | **35.0%** |

The final method gives about **5×** the Kendall τ of the TF-IDF baseline, **1.7×** its rate of correct adjacent pairs, and nearly **4×** its rate of correct opening sentences.

---

## Task 3: Translation to English

**Model:** [`ai4bharat/indictrans2-indic-en-dist-200M`](https://huggingface.co/ai4bharat/indictrans2-indic-en-dist-200M), used as released with no fine-tuning, together with `IndicTransToolkit` for pre- and post-processing.

**Pipeline**
1. Each 5-sentence group is translated as **one unit**, so the model sees the surrounding context.
2. If a group exceeds the model's 256-token window, it is split at sentence boundaries, and the pieces are translated and joined back together.
3. Beam search with 5 beams, fp16.
4. **Automatic quality checks:** any output that hits the length limit, still contains Devanagari, or has an unusual English/Hindi length ratio is re-translated sentence by sentence, and the better version is kept.

**Results**
- **FLORES-200 devtest** (200 sentences, Hindi → English): **BLEU 46.27, chrF++ 68.11**
- **Samples:** 91 groups across the 7 samples translated, with every group passing the quality checks.

Example (sample-1, group 2):
> **Hindi:** दक्षिण रेलवे के चेन्नै स्टेशन से धनुष्कोटि जाने वाली मुख्य लाइन पर चेन्नै से 35 मील की दूरी पर चेंगलपट स्टेशन है। चेंगलपट स्टेशन से एक लाइन अरकोनम तक जाती है। …
>
> **English:** Chengalpattu station is 35 miles from Chennai on the main line of Southern Railway from Chennai station to Dhanushkoti. A line from Chengalpattu station goes to Arakkonam. …

---

## Repository structure

```
├── README.md
├── notebooks/
│   ├── Task1_Hindi_Punctuation.ipynb
│   ├── Task2_Sentence_Ordering.ipynb
│   └── Task3_Translation.ipynb
└── outputs/
    ├── task1/   punctuated/sample-*.txt, eval_report.txt, memory_report.json
    ├── task2/   ordered/, groups/, all_groups.json, eval_task2.txt, report_task2.json
    └── task3/   english/, parallel/, all_translations.csv, report_task3.json
```

## How to run

All notebooks run on **Kaggle** with GPU T4 and Internet enabled.

1. **Task 1:** add the raw samples as a Kaggle dataset, then run all cells. Produces the punctuated samples and Model 1.
2. **Task 2:** add the Task 1 notebook output as input, then run all cells. Produces the ordered groups and Model 2.
3. **Task 3:** add the Task 2 notebook output as input, plus a Hugging Face token as the Kaggle secret `HF_TOKEN` (IndicTrans2 is gated). Run CELL 1 first, because it installs `transformers<5`, which IndicTrans2 requires. Then run the remaining cells.

## Main libraries

PyTorch · Hugging Face Transformers · Datasets · scikit-learn · IndicTransToolkit · sacreBLEU

## References

- AI4Bharat Sangraha corpus: Khan et al., 2024
- IndicTrans2: Gala et al., *TMLR* 2023
- BERT / multilingual BERT: Devlin et al., 2019
- DistilBERT: Sanh et al., 2019
- FLORES-200: NLLB Team, 2022

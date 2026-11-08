# Data

## `raw/`: benchmark files exactly as used

Downloaded once by `src/tokens_per_task/datasets.py` and committed unmodified, so the item samples can be reproduced without network access. Licence texts and attribution: [`raw/THIRD_PARTY_NOTICES.md`](raw/THIRD_PARTY_NOTICES.md).

| File | Dataset and source | Licence | SHA-256 |
|---|---|---|---|
| `banking77_test.csv` | **Banking77** test split, 3,080 customer messages, 77 intents. [PolyAI-LDN/task-specific-datasets](https://github.com/PolyAI-LDN/task-specific-datasets) | CC BY 4.0 | `d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d` |
| `gsm8k_test.jsonl` | **GSM8K** test split, 1,319 problems. [openai/grade-school-math](https://github.com/openai/grade-school-math) | MIT, © 2021 OpenAI | `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14` |
| `math500_test.jsonl` | **MATH-500**, 500 problems (74 are level 5 with an integer answer). Downloaded from [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500); originates in [openai/prm800k](https://github.com/openai/prm800k), drawn from [MATH](https://github.com/hendrycks/math) | MIT, © 2023 OpenAI (PRM800K) and © 2021 Dan Hendrycks (MATH)¹ | `35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132` |

¹ The Hugging Face mirror states no licence of its own; the licences are those of its upstream sources.

Item samples: `random.Random(42).sample(rows, n)` with n = 100 (T1), 100 (T2), 50 (T4). Item ids encode the row index in the file (`b77-0102`, `gsm-0123`, `math-042`).

### References

If you use these datasets, cite the original work:

- **Banking77:** I. Casanueva, T. Temčinas, D. Gerz, M. Henderson, I. Vulić. Efficient Intent Detection with Dual Sentence Encoders. *Proceedings of the 2nd Workshop on NLP for ConvAI, ACL 2020.* [arXiv:2003.04807](https://arxiv.org/abs/2003.04807)
- **GSM8K:** K. Cobbe, V. Kosaraju, M. Bavarian, et al. Training Verifiers to Solve Math Word Problems. 2021. [arXiv:2110.14168](https://arxiv.org/abs/2110.14168)
- **MATH:** D. Hendrycks, C. Burns, S. Kadavath, et al. Measuring Mathematical Problem Solving With the MATH Dataset. *NeurIPS Datasets and Benchmarks*, 2021. [arXiv:2103.03874](https://arxiv.org/abs/2103.03874)
- **MATH-500:** H. Lightman, V. Kosaraju, Y. Burda, et al. Let's Verify Step by Step. *ICLR*, 2024. [arXiv:2305.20050](https://arxiv.org/abs/2305.20050)

BibTeX for all of them: [`paper/references.bib`](../paper/references.bib).

## `generated/`: the synthetic T3 ledger

Written by `python run.py check` from seed 42; byte-identical on every run. Created for this study; no third-party data.

| File | Contents | SHA-256 |
|---|---|---|
| `t3_ledger.json` | 12 fictitious suppliers, 500 invoices dated 2025 (EUR) | `eba6e16f03544dd385f8e662e3946dd5ec2d878c5cd7f6fccd209bfaa4b2ed54` |
| `t3_questions.json` | 20 questions with answer keys, parameters and the expected number of tool calls | `caaf3a1993c9a4b91fee0f610b02e8ddb04510bbc6805c7786083467eb476db7` |

All supplier names are invented. Released under CC BY 4.0.

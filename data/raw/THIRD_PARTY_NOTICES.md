# Third-party data notices

The files in this folder are unmodified copies of third-party datasets, redistributed under
their original licences. Only random samples drawn from them (seed 42) were used in this study.

---

## banking77_test.csv: Banking77 (test split)

- **Creators:** Iñigo Casanueva, Tadas Temčinas, Daniela Gerz, Matthew Henderson, Ivan Vulić (PolyAI)
- **Source:** https://github.com/PolyAI-LDN/task-specific-datasets (`banking_data/test.csv`)
- **Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0),
  https://creativecommons.org/licenses/by/4.0/
- **Changes:** none (file redistributed as downloaded).
- **Requested citation:** Casanueva et al., "Efficient Intent Detection with Dual Sentence
  Encoders", Proceedings of the 2nd Workshop on NLP for ConvAI, ACL 2020.
  https://arxiv.org/abs/2003.04807

---

## gsm8k_test.jsonl: GSM8K (test split)

- **Creators:** Karl Cobbe et al. (OpenAI)
- **Source:** https://github.com/openai/grade-school-math (`grade_school_math/data/test.jsonl`)
- **Citation:** Cobbe et al., "Training Verifiers to Solve Math Word Problems", arXiv:2110.14168, 2021.
- **Changes:** none.
- **Licence:**

```
MIT License

Copyright (c) 2021 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## math500_test.jsonl: MATH-500

- **What it is:** the 500-problem test subset of the MATH benchmark selected by OpenAI for
  "Let's Verify Step by Step" (PRM800K).
- **Downloaded from:** https://huggingface.co/datasets/HuggingFaceH4/MATH-500 (`test.jsonl`).
  The Hugging Face dataset card states no licence of its own; it points to the PRM800K
  repository as the source.
- **Upstream sources and licences:**
  - PRM800K, https://github.com/openai/prm800k: MIT License, Copyright (c) 2023 OpenAI.
  - MATH, https://github.com/hendrycks/math: MIT License, Copyright (c) 2021 Dan Hendrycks.
- **Citations:** Hendrycks et al., "Measuring Mathematical Problem Solving With the MATH
  Dataset", NeurIPS Datasets and Benchmarks, 2021, arXiv:2103.03874. Lightman et al., "Let's
  Verify Step by Step", ICLR 2024, arXiv:2305.20050.
- **Changes:** none to the file. The harness uses only level-5 problems with integer answers.
- **Licences** (MIT; the permission text is identical to the GSM8K notice above):

```
MIT License

Copyright (c) 2023 OpenAI
```

```
MIT License

Copyright (c) 2021 Dan Hendrycks
```

Permission is hereby granted, free of charge, to any person obtaining a copy of this software
and associated documentation files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions: The above copyright
notice and this permission notice shall be included in all copies or substantial portions of
the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.

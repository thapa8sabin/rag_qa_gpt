# 🔎 Feasibility Study, Limitations & Real-World Usefulness

This document evaluates how practical Quiz-GPT is today, where it currently falls short, and the concrete value it can already deliver. It also outlines promising future directions — including domain adaptation for government examination systems such as Nepal’s Lok Sewa.

---

## 1. Feasibility Study

### 1.1 Technical Feasibility — ✅ High

| Aspect                      | Assessment                             | Notes                                                                                                |
| --------------------------- | -------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| Local execution             | Fully feasible                         | Runs on CPU; GPU optional but beneficial                                                             |
| Dependency stack            | Mature and well-supported              | uv, FastMCP, Strands, Transformers, Qdrant, Ollama                                                   |
| Question generation quality | Good for factual / educational content | T5-based generator + ranking works well on clear prose                                               |
| Custom fine-tuning          | Feasible but time-consuming            | Golden-only is manageable; mixing large public datasets significantly increases training time on CPU |
| End-to-end workflow         | Complete                               | Ingest → Retrieve → Generate → Rank → Store → Chat                                                   |

The system is already usable for personal study tools, internal knowledge-base quizzes, and small-scale educational platforms.

### 1.2 Domain Adaptation Feasibility — ✅ Promising

One of the strongest design decisions is the **fine-tunable QA Evaluator**. Because the ranking model can be retrained on domain-specific golden data, the system can be specialized without changing the generation architecture.

#### Suggested direction: Nepali Government Knowledge Banks (Lok Sewa & similar)

> **Idea**: Use models that support Nepali (or multilingual models) together with official question banks, FAQs, and syllabus materials from Lok Sewa Aayog (Public Service Commission of Nepal) and related government bodies as a **golden dataset**. Fine-tune both the evaluator and, in a later stage, explore generation models that handle Nepali better. The resulting custom model could generate high-quality practice questions that closely match the style and content of real examinations.

**Why this is feasible**

- Lok Sewa and similar commissions publish large volumes of past questions, syllabi, and FAQs — ideal golden data.
- Multilingual models (or Nepali-specific models such as those based on mT5, IndicBERT, or newer Nepali LLMs) can be substituted or used for embedding / generation.
- The existing training pipeline already supports:
  - Loading a custom golden JSON
  - Upsampling domain examples
  - Mixing with public data for robustness
- Once fine-tuned, the evaluator improves ranking of generated questions for that exact domain.

**Possible implementation path**

1. Collect and clean Lok Sewa past papers + official FAQs → convert into the golden dataset format (`question`, `context`, `ideal_answer`).
2. Ingest the same materials (and additional textbooks / acts) into Qdrant.
3. Fine-tune the QA Evaluator using the golden set (high upsample factor).
4. Optionally replace or augment the T5 generator with a multilingual / Nepali-capable sequence-to-sequence model.
5. Deploy as a practice platform for aspirants.

This approach is realistic for a research prototype or a focused product for civil-service exam preparation.

### 1.3 Suggested Additional Feasibility Experiments

| Experiment                              | Goal                                        | Expected effort |
| --------------------------------------- | ------------------------------------------- | --------------- |
| Nepali / multilingual generator         | Better handling of non-English content      | Medium          |
| Larger / newer ranking models           | Higher quality filtering                    | Low–Medium      |
| Active learning loop                    | Let users mark good/bad questions → retrain | Medium          |
| Multi-hop / reasoning questions         | Move beyond single-sentence answers         | High            |
| Integration with official LMS platforms | Real institutional deployment               | High            |

---

## 2. Current Limitations

| Limitation                                   | Impact                                                                                                  | Mitigation / Future work                                                                                   |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| Generator is English-centric                 | Weaker results on Nepali / other languages                                                              | Swap in multilingual T5 / mT5 / local models                                                               |
| Relies on sentence-level answers             | Less suitable for complex reasoning questions                                                           | Explore answer-span or free-form generation                                                                |
| Golden dataset size                          | Domain adaptation quality depends on data volume                                                        | Crowdsource or extract from past papers                                                                    |
| No built-in multiple-choice generation       | User must post-process or prompt the agent                                                              | Add a dedicated MCQ generation step                                                                        |
| CPU inference can be slow                    | Longer wait times for large texts                                                                       | Quantization, smaller models, or GPU                                                                       |
| Evaluation is binary / ranking only          | Does not measure factual correctness against source                                                     | Add entailment or attribution checks                                                                       |
| Limited handling of tables / structured data | Knowledge base is currently free text only                                                              | Extend ingestion to structured sources                                                                     |
| Custom fine-tuning is slow with public data  | Training becomes very long when large public datasets are mixed with the golden set (especially on CPU) | Train primarily on golden data, use smaller public samples, or move to GPU; support resume / checkpointing |

These limitations are typical of research-stage educational AI systems and do not prevent useful application in the target domains.

---

## 3. Real-World Usefulness

### 3.1 Immediate value

- **Personal study assistants** — Students can turn any set of notes or textbooks into practice quizzes.
- **Teacher / instructor tooling** — Quickly generate formative assessment questions from lecture material.
- **Internal knowledge bases** — Companies or NGOs can create quizzes from policy documents, manuals, or FAQs.
- **Exam preparation platforms** — Especially powerful when fine-tuned on past papers (as proposed for Lok Sewa).

### 3.2 Longer-term potential

| Domain                                     | How Quiz-GPT helps                                                             |
| ------------------------------------------ | ------------------------------------------------------------------------------ |
| Civil service exams (Lok Sewa, UPSC, etc.) | Domain-adapted question generation + ranking from official materials           |
| Medical / nursing education                | Generate questions from clinical guidelines and textbooks                      |
| Corporate training                         | Turn internal documentation into onboarding quizzes                            |
| Language learning                          | Produce comprehension questions from graded readers                            |
| Public sector knowledge management         | Make government circulars and acts more accessible through interactive quizzes |

## 4. Recommendations & Future Improvements

1. **Short term**
   - Expand the golden dataset with high-quality educational material.
   - Run the existing fine-tuning pipeline (preferably golden-heavy or on GPU) and measure ranking improvement.
   - Add simple multiple-choice option generation on top of the current open-ended answers.

2. **Medium term**
   - Prototype the Lok Sewa / Nepali use-case: collect past questions, fine-tune the evaluator, and evaluate with human raters.
   - Experiment with multilingual generation models.
   - Add a lightweight human-in-the-loop feedback mechanism.
   - **Study-pack generation**: After retrieving context, automatically produce a complete study pack that includes:
     - A concise summary of the source material
     - Ranked question–answer pairs
     - Optional multiple-choice versions
     - Key terms / flashcards  
       This turns a single topic query into a ready-to-use learning unit.

3. **Long term**
   - Explore generation of multi-hop and higher-order thinking questions.
   - Integrate with learning-management systems or spaced-repetition tools.
   - Consider distillation or quantization for mobile / low-resource deployment.
   - Further reduce fine-tuning time through better sampling strategies, parameter-efficient methods (LoRA / QLoRA), and resume-from-checkpoint support.

---

## 5. Conclusion

Quiz-GPT is **technically feasible and already useful** for generating practice quizzes from textual knowledge bases. Its most distinctive strength is the ability to fine-tune the quality evaluator on domain-specific golden data — opening a clear path toward specialized applications such as Nepali civil-service exam preparation.

While limitations around language coverage, question complexity, and scale remain, none of them are fundamental blockers. With focused data collection and modest model upgrades, the system can deliver tangible value in education, training, and public-sector knowledge dissemination.

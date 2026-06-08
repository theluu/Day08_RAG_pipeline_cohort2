# Bài Tập Nhóm — Search Engine / RAG Chatbot

## Mục Tiêu

Sau khi hoàn thành bài cá nhân, nhóm ngồi lại để xây dựng **1 trong 2 sản phẩm**:

---

## Yêu cầu 1:  Sản phẩm nhóm RAG Chatbot

Xây dựng chatbot trả lời câu hỏi về pháp luật ma tuý và tin tức liên quan.

**Yêu cầu:**
- Giao diện chat (Streamlit / Gradio / Chainlit)
- Trả lời có citation (dựa trên Task 10)
- Hỗ trợ follow-up questions (conversation memory)
- Hiển thị source documents đã dùng

**Stack gợi ý:**
```
Chainlit/Streamlit → Retrieval (Task 9) → Generation (Task 10) → Display
```

---

## Yêu cầu 2: RAG Evaluation Pipeline

Sử dụng **1 trong 3 framework** sau để evaluate pipeline RAG của nhóm:

### Framework lựa chọn

| Framework | Cài đặt | Đặc điểm |
|-----------|---------|-----------|
| [DeepEval](https://github.com/confident-ai/deepeval) | `pip install deepeval` | Nhiều metric built-in, dễ integrate với pytest |
| [RAGAS](https://github.com/explodinggradients/ragas) | `pip install ragas` | Chuẩn industry cho RAG eval, 3 trục chính |
| [TruLens](https://github.com/truera/trulens) | `pip install trulens` | Dashboard UI, feedback functions mạnh |

### Yêu cầu Evaluation

1. **Tạo Golden Dataset** — tối thiểu 15 cặp Q&A (question, expected_answer, expected_context)
2. **Chạy evaluation** trên toàn bộ golden dataset với các metrics sau:
   - **Faithfulness** — câu trả lời có bám đúng context không?
   - **Answer Relevance** — câu trả lời có đúng câu hỏi không?
   - **Context Recall** — retriever có lấy đủ evidence không?
   - **Context Precision** — trong context lấy về, bao nhiêu % thực sự hữu ích?
3. **So sánh A/B** — chạy eval trên ít nhất 2 config khác nhau (ví dụ: có reranking vs không reranking, hoặc hybrid vs dense-only)
4. **Báo cáo** — bảng điểm + phân tích worst performers + đề xuất cải tiến

### Code mẫu — DeepEval

```python
from deepeval import evaluate
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
)
from deepeval.test_case import LLMTestCase

# Tạo test cases từ golden dataset
test_cases = []
for item in golden_dataset:
    result = rag_pipeline.generate_with_citation(item["question"])
    test_case = LLMTestCase(
        input=item["question"],
        actual_output=result["answer"],
        expected_output=item["expected_answer"],
        retrieval_context=[c["content"] for c in result["sources"]],
    )
    test_cases.append(test_case)

# Chạy evaluation
metrics = [
    FaithfulnessMetric(threshold=0.7),
    AnswerRelevancyMetric(threshold=0.7),
    ContextualRecallMetric(threshold=0.7),
    ContextualPrecisionMetric(threshold=0.7),
]

results = evaluate(test_cases, metrics)
```

### Code mẫu — RAGAS

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from datasets import Dataset

# Chuẩn bị data
eval_data = {
    "question": [],
    "answer": [],
    "contexts": [],
    "ground_truth": [],
}

for item in golden_dataset:
    result = rag_pipeline.generate_with_citation(item["question"])
    eval_data["question"].append(item["question"])
    eval_data["answer"].append(result["answer"])
    eval_data["contexts"].append([c["content"] for c in result["sources"]])
    eval_data["ground_truth"].append(item["expected_answer"])

dataset = Dataset.from_dict(eval_data)

# Chạy evaluation
result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
)
print(result.to_pandas())
```

### Code mẫu — TruLens

```python
from trulens.apps.custom import TruCustomApp, instrument
from trulens.core import Feedback
from trulens.providers.openai import OpenAI as TruOpenAI

provider = TruOpenAI()

# Define feedback functions
f_faithfulness = Feedback(provider.groundedness_measure_with_cot_reasons).on_output()
f_relevance = Feedback(provider.relevance).on_input_output()
f_context_relevance = Feedback(provider.context_relevance).on_input()

# Wrap RAG pipeline
tru_rag = TruCustomApp(
    rag_pipeline,
    app_name="DrugLaw_RAG",
    feedbacks=[f_faithfulness, f_relevance, f_context_relevance],
)

# Run evaluation
with tru_rag as recording:
    for item in golden_dataset:
        rag_pipeline.generate_with_citation(item["question"])

# View dashboard
from trulens.dashboard import run_dashboard
run_dashboard()
```

### Deliverable Evaluation

- [x] File `group_project/evaluation/golden_dataset.json` — 15+ cặp Q&A
- [x] File `group_project/evaluation/eval_pipeline.py` — script chạy evaluation
- [x] File `group_project/evaluation/results.md` — bảng điểm + phân tích
- [x] So sánh A/B ít nhất 2 configs

### Yêu Cầu 2 — Evaluation Pipeline

Implemented in `group_project/evaluation/eval_pipeline.py`.

Framework choice: custom deterministic evaluator. It keeps the required four
metrics without adding a separate judge dependency:

- Faithfulness: answer terms grounded in retrieved context.
- Answer Relevance: expected answer coverage in actual answer.
- Context Recall: expected evidence hints found in retrieved context.
- Context Precision: fraction of retrieved chunks containing expected hints.

A/B configs:

- Config A `hybrid_rerank`: Task 9 full pipeline + Task 10 generation.
- Config B `lexical_only`: Task 6 BM25 baseline + Task 10 generation.

Run:

```bash
.venv311/bin/python group_project/evaluation/eval_pipeline.py
```

Latest report: `group_project/evaluation/results.md`.

---

## Yêu Cầu Chung

| Yêu cầu | Trạng thái | Minh chứng |
|---------|------------|------------|
| Tích hợp pipeline từ bài cá nhân | Done | `group_project/app.py` gọi Task 9 `retrieve()` và Task 10 `generate_with_citation()` |
| Demo hoạt động được trong buổi trình bày | Done | Streamlit app chạy local bằng lệnh `streamlit run group_project/app.py` |
| Evaluation pipeline chạy được và có báo cáo kết quả | Done | `group_project/evaluation/eval_pipeline.py` đã chạy và xuất `group_project/evaluation/results.md` |
| README mô tả kiến trúc | Done | Phần "Kiến Trúc Hệ Thống" bên dưới mô tả Streamlit -> Retrieval -> Generation -> Display |

### Demo Checklist

- Chat UI: `group_project/app.py`
- Conversation memory: recent chat turns được thêm vào retrieval query.
- Citation: Task 10 sinh câu trả lời có citation dạng `[Nguồn, Năm]`.
- Source display: mỗi câu trả lời hiển thị chunks, score, route và metadata.
- Evaluation report: `group_project/evaluation/results.md`.

---

## Kiến Trúc Hệ Thống

```
Streamlit Chat UI (group_project/app.py)
  -> Conversation memory contextualizes follow-up questions
  -> Task 9 retrieve()
       -> Task 5 semantic search
       -> Task 6 BM25 lexical search
       -> Task 7 RRF merge + rerank
       -> Task 8 PageIndex fallback when hybrid score is weak
  -> Task 10 generate_with_citation()
       -> lost-in-the-middle context reordering
       -> OpenAI answer generation with [Nguồn, Năm] citations
  -> Display answer + source chunks + metadata
```

### Yêu Cầu 1 — RAG Chatbot

Implemented in `group_project/app.py`.

| Yêu cầu | Trạng thái | Ghi chú |
|---------|------------|---------|
| Giao diện chat | Done | Streamlit `st.chat_message` + `st.chat_input` |
| Trả lời có citation | Done | Dùng Task 10, citation dạng `[Nguồn, Năm]` |
| Follow-up questions | Done | Recent conversation turns được inject vào retrieval query |
| Hiển thị source documents | Done | Mỗi chunk có content, score, route, metadata |
| Pipeline tích hợp | Done | Streamlit -> Task 9 -> Task 10 -> Display |

---

## Phân Công Công Việc

| Thành viên | MSSV | Nhiệm vụ | Trạng thái |
|-----------|------|----------|------------|
| | | | |
| | | | |
| | | | |
| | | | |

---

## Hướng Dẫn Chạy

```bash
# Cài đặt dependencies
pip install -r requirements.txt

# Chạy app
streamlit run group_project/app.py
```

Nếu dùng virtualenv local của repo:

```bash
.venv311/bin/streamlit run group_project/app.py --server.port 8502 --server.address 127.0.0.1
```

Mở trình duyệt tại `http://127.0.0.1:8502`.

---

## Lưu ý: Hãy giữ lại repo này nếu như bạn học track 3 giai đoạn 2, chúng ta sẽ phát triển tiếp dự án lên knowledge graph để khắc phục các câu hỏi hóc búa khi có các câu hỏi khó.

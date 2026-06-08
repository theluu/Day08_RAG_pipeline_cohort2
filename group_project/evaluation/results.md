# RAG Evaluation Results

## Framework sử dụng

Custom deterministic evaluator, implemented in `group_project/evaluation/eval_pipeline.py`.
The evaluator measures the four required RAG dimensions without extra judge dependencies.

## Dataset

- Golden dataset size: 15 Q&A pairs
- Domain: Vietnamese drug law and related news

## Overall Scores

| Metric | Config A: hybrid + rerank | Config B: lexical-only | Δ |
|--------|---------------------------|------------------------|---|
| Faithfulness | 0.7915 | 0.7902 | +0.0013 |
| Answer Relevance | 0.6463 | 0.6672 | -0.0209 |
| Context Recall | 0.9256 | 0.9256 | +0.0000 |
| Context Precision | 0.8800 | 0.8800 | +0.0000 |
| Average | 0.8109 | 0.8157 | -0.0048 |

## A/B Comparison Analysis

**Config A: hybrid_rerank**

Uses Task 9 full retrieval: semantic search + BM25 lexical search, RRF merge, rerank, and PageIndex fallback when available.

**Config B: lexical_only**

Uses only Task 6 BM25 chunks, then the same Task 10 citation generator. This isolates the value of hybrid retrieval and reranking.

**Kết luận:** `lexical_only` has the higher average score in this run. Use the per-case table below to inspect whether errors come from retrieval coverage or generation grounding.

## Worst Performers (Bottom 3, Config A)

| # | Question | Faithfulness | Relevance | Recall | Precision | Failure Stage | Root Cause |
|---|----------|--------------|-----------|--------|-----------|---------------|------------|
| 1 | Điều 248 Bộ luật Hình sự quy định gì về tội sản xuất trái phép chất ma túy? | 0.0000 | 0.0000 | 0.3333 | 1.0000 | retrieval | Retriever missed expected evidence |
| 2 | Bài về ca sĩ Miu Lê bị bắt ở bãi biển nêu cáo buộc gì? | 0.0000 | 0.0000 | 1.0000 | 1.0000 | generation | Answer does not cover expected answer |
| 3 | Tiền chất theo Luật Phòng, chống ma túy 2021 được hiểu là gì? | 0.7500 | 0.3000 | 1.0000 | 0.4000 | generation | Answer does not cover expected answer |

## Per-Case Scores

| Config | ID | Avg | Faithfulness | Relevance | Recall | Precision | Sources |
|--------|----|-----|--------------|-----------|--------|-----------|---------|
| hybrid_rerank | legal_001 | 0.8934 | 0.9167 | 0.8571 | 1.0000 | 0.8000 | bo-luat-hinh-su-2015-sua-doi-2017.md, nghi-dinh-105-2021.md |
| hybrid_rerank | legal_002 | 0.3333 | 0.0000 | 0.0000 | 0.3333 | 1.0000 | nghi-dinh-105-2021.md |
| hybrid_rerank | legal_003 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | nghi-dinh-105-2021.md, luat-phong-chong-ma-tuy-2021.md |
| hybrid_rerank | legal_004 | 0.9500 | 1.0000 | 1.0000 | 1.0000 | 0.8000 | luat-phong-chong-ma-tuy-2021.md, nghi-dinh-105-2021.md |
| hybrid_rerank | legal_005 | 0.9606 | 0.9600 | 0.8824 | 1.0000 | 1.0000 | luat-phong-chong-ma-tuy-2021.md, bo-luat-hinh-su-2015-sua-doi-2017.md |
| hybrid_rerank | legal_006 | 0.9179 | 0.9714 | 0.7000 | 1.0000 | 1.0000 | nghi-dinh-105-2021.md |
| hybrid_rerank | legal_007 | 0.6125 | 0.7500 | 0.3000 | 1.0000 | 0.4000 | luat-phong-chong-ma-tuy-2021.md, nghi-dinh-105-2021.md |
| hybrid_rerank | legal_008 | 0.8893 | 0.8696 | 0.6875 | 1.0000 | 1.0000 | luat-phong-chong-ma-tuy-2021.md |
| hybrid_rerank | news_001 | 0.8620 | 0.8800 | 0.8182 | 0.7500 | 1.0000 | phuc-xo-va-11-nguoi-bi-bat-tam-giam-bao-vnexpress.md |
| hybrid_rerank | news_002 | 0.8942 | 0.8767 | 0.9000 | 0.8000 | 1.0000 | phuc-xo-va-11-nguoi-bi-bat-tam-giam-bao-vnexpress.md, phuc-xo-va-em-trai-bi-bat-khan-cap-bao-vnexpress.md |
| hybrid_rerank | news_003 | 0.7494 | 0.8644 | 0.5333 | 1.0000 | 0.6000 | ma-tuy-trong-loi-song-showbiz-bao-vnexpress.md, bat-ca-si-long-nhat-va-ca-si-son-ngoc-minh-vi-lien-quan-ma-tuy-tuoi-tr.md |
| hybrid_rerank | news_004 | 0.7688 | 0.8750 | 0.6000 | 1.0000 | 0.6000 | nhung-nghe-si-viet-nga-ngua-vi-ma-tuy-ngoi-sao.md, ma-tuy-trong-loi-song-showbiz-bao-vnexpress.md |
| hybrid_rerank | news_005 | 0.9375 | 1.0000 | 0.7500 | 1.0000 | 1.0000 | ca-si-chau-viet-cuong-bi-e-nghi-13-14-nam-tu.md, ca-si-chau-viet-cuong-hau-toa-vi-nhet-toi-hai-chet-co-gai-20-tuoi.md |
| hybrid_rerank | news_006 | 0.5000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | ca-si-miu-le-bi-bat-qua-tang-dung-ma-tuy-o-bai-bien.md, ca-si-miu-le-bi-bat-voi-cao-buoc-to-chuc-su-dung-ma-tuy-bao-vnexpress.md |
| hybrid_rerank | news_007 | 0.8940 | 0.9091 | 0.6667 | 1.0000 | 1.0000 | bat-ca-si-long-nhat-va-ca-si-son-ngoc-minh-vi-lien-quan-ma-tuy-tuoi-tr.md |
| lexical_only | legal_001 | 0.8937 | 0.9178 | 0.8571 | 1.0000 | 0.8000 | bo-luat-hinh-su-2015-sua-doi-2017.md, nghi-dinh-105-2021.md |
| lexical_only | legal_002 | 0.3333 | 0.0000 | 0.0000 | 0.3333 | 1.0000 | nghi-dinh-105-2021.md |
| lexical_only | legal_003 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | nghi-dinh-105-2021.md, luat-phong-chong-ma-tuy-2021.md |
| lexical_only | legal_004 | 0.9500 | 1.0000 | 1.0000 | 1.0000 | 0.8000 | luat-phong-chong-ma-tuy-2021.md, nghi-dinh-105-2021.md |
| lexical_only | legal_005 | 0.9532 | 0.9306 | 0.8824 | 1.0000 | 1.0000 | luat-phong-chong-ma-tuy-2021.md, bo-luat-hinh-su-2015-sua-doi-2017.md |
| lexical_only | legal_006 | 0.9091 | 0.9365 | 0.7000 | 1.0000 | 1.0000 | nghi-dinh-105-2021.md |
| lexical_only | legal_007 | 0.6162 | 0.7647 | 0.3000 | 1.0000 | 0.4000 | luat-phong-chong-ma-tuy-2021.md, nghi-dinh-105-2021.md |
| lexical_only | legal_008 | 0.9844 | 0.9375 | 1.0000 | 1.0000 | 1.0000 | luat-phong-chong-ma-tuy-2021.md |
| lexical_only | news_001 | 0.8615 | 0.8776 | 0.8182 | 0.7500 | 1.0000 | phuc-xo-va-11-nguoi-bi-bat-tam-giam-bao-vnexpress.md |
| lexical_only | news_002 | 0.8917 | 0.8667 | 0.9000 | 0.8000 | 1.0000 | phuc-xo-va-11-nguoi-bi-bat-tam-giam-bao-vnexpress.md, phuc-xo-va-em-trai-bi-bat-khan-cap-bao-vnexpress.md |
| lexical_only | news_003 | 0.7494 | 0.8644 | 0.5333 | 1.0000 | 0.6000 | ma-tuy-trong-loi-song-showbiz-bao-vnexpress.md, bat-ca-si-long-nhat-va-ca-si-son-ngoc-minh-vi-lien-quan-ma-tuy-tuoi-tr.md |
| lexical_only | news_004 | 0.7621 | 0.8485 | 0.6000 | 1.0000 | 0.6000 | nhung-nghe-si-viet-nga-ngua-vi-ma-tuy-ngoi-sao.md, ma-tuy-trong-loi-song-showbiz-bao-vnexpress.md |
| lexical_only | news_005 | 0.9375 | 1.0000 | 0.7500 | 1.0000 | 1.0000 | ca-si-chau-viet-cuong-bi-e-nghi-13-14-nam-tu.md, ca-si-chau-viet-cuong-hau-toa-vi-nhet-toi-hai-chet-co-gai-20-tuoi.md |
| lexical_only | news_006 | 0.5000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | ca-si-miu-le-bi-bat-qua-tang-dung-ma-tuy-o-bai-bien.md, ca-si-miu-le-bi-bat-voi-cao-buoc-to-chuc-su-dung-ma-tuy-bao-vnexpress.md |
| lexical_only | news_007 | 0.8940 | 0.9091 | 0.6667 | 1.0000 | 1.0000 | bat-ca-si-long-nhat-va-ca-si-son-ngoc-minh-vi-lien-quan-ma-tuy-tuoi-tr.md |

## Recommendations

### Cải tiến 1
**Action:** Clean markdown boilerplate from crawled news pages before chunking.
**Expected impact:** Higher context precision and fewer irrelevant chunks.

### Cải tiến 2
**Action:** Add metadata fields such as publication date, source platform, and legal article number during conversion.
**Expected impact:** Better citation quality and easier source filtering.

### Cải tiến 3
**Action:** Increase PageIndex credits or provide `PAGEINDEX_DOC_IDS` for uploaded legal PDFs.
**Expected impact:** Better fallback coverage when hybrid retrieval score is weak.

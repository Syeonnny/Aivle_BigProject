# 📑 육아 도우미 챗봇 개발 프로젝트
이 저장소는 2025년 1월부터 2월까지 약 4주간 진행된 LLM 기반 RAG 챗봇 개발 프로젝트의 결과물을 포함하고 있습니다. 본 프로젝트는 스마트 패치를 활용한 육아 도우미 앱에 탑재될 챗봇을 개발하기 위한 실험 및 구현 과정을 담고 있으며, 특히 정부의 공신력 있는 문서 기반 RAG 모델을 활용하여 신뢰성 높은 답변을 제공하는 것을 목표로 진행되었습니다.

## 🗂 프로젝트 개요
본 저장소는 만 3세 이하 영유아 부모를 위한 육아 도우미 챗봇 개발을 중심으로, FAISS + Cohere Reranking 하이브리드 검색, GPT-4o-mini, LangChain 기반 RAG 파이프라인 등을 활용하여 고품질 응답을 생성하는 구조로 설계되었습니다. 또한, 개발 과정에서는 UI/UX 협업, CI/CD 자동화, 평가 지표 기반의 성능 개선 등 다양한 실무 요소를 함께 반영하였습니다.

## 📌 주요 구현 및 성과
### 1. LLM 기반 RAG 챗봇 개발 및 최적화
- 정부 문서 총 11,563문장을 기반으로 FAISS 벡터 검색 + Cohere Reranker를 활용한 하이브리드 검색 구현
- OpenAI GPT-4o-mini와 LangChain RAG 파이프라인을 활용하여 정확도 및 신뢰성 향상

### 2. 성능 개선 및 정량 평가
- RAGAS 평가 결과
  - Faithfulness: 0.88
  - Answer Relevancy: 0.85
  - BERT Score: 0.92 → 검색 정확도 및 응답 품질 약 10% 개선
- BM25 대비 FAISS 검색 속도 1.5배 향상

### 3. 협업 및 개발 환경 고도화
- Figma 기반 UI/UX 협업, 애자일(Sprint) 방식 적용
- GitHub 기반 CI/CD 자동화, 코드 리뷰를 통한 품질 유지
- 실제 서비스 수준의 신뢰성 있는 챗봇 구축

---

## 📈 주요 기술 스택 및 구조
- LLM: OpenAI GPT-4o-mini
- 검색 시스템: FAISS + Cohere Reranker
- RAG 프레임워크: LangChain
- 평가 도구: RAGAS, BERT Score
- 프론트/협업 도구: Figma, GitHub, Agile Sprint

---

## 🚀 향후 계획
- 사용자 피드백 기반 응답 개선 알고리즘 고도화
- 다국어 대응 가능 확장
- LangGraph 기반 대화 흐름 제어 구조 실험 및 적용

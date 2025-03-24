import os
import json
from pdf_loader import process_pdf_folder, process_ocr_data, create_json_from_data
from chunk_embedding import FAISSRetrieverWithCohere, initialize_retriever
from rag import create_rag_chain, get_rag_response
from langchain_upstage import UpstageEmbeddings
from dotenv import load_dotenv
import faiss


# 환경 변수 로드
load_dotenv("chat_api_key.env")
# 주요 설정
PDF_FOLDER = "./data/pdf_files"
OCR_FOLDER = "./data/ocr_files"
OUTPUT_JSON = "./data/pdf_load_data.json"
FAISS_INDEX = "./data/faiss_index.bin"
UPSTAGE_API_KEY = os.getenv("embedding_api_key")
OPENAI_API_KEY = os.getenv("openai_api_key")
COHERE_API_KEY = os.getenv("cohere_api_key")


def check_faiss_index(index_path):
    """FAISS 인덱스 파일의 존재 여부를 확인하고 로드"""
    print(f"FAISS 인덱스 파일 경로: {index_path}")
    if os.path.exists(index_path):
        print("FAISS 인덱스 파일이 존재합니다.")
        index = faiss.read_index(index_path)
        print(f"FAISS 인덱스 로드 완료. 벡터 개수: {index.ntotal}")
        return index
    else:
        print("Error: FAISS 인덱스 파일이 존재하지 않습니다.")
        return None

def main():
    # PDF 및 OCR 데이터 처리
    pdf_data = process_pdf_folder(PDF_FOLDER)
    ocr_data = process_ocr_data(OCR_FOLDER)
    create_json_from_data(pdf_data, ocr_data, OUTPUT_JSON)
    print(f"JSON 파일 저장 완료: {OUTPUT_JSON}")

    # JSON 데이터 로드 텍스트 청크 생성
    with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
        combined_data = json.load(f)

    # 텍스트 청크 생성
    chunks = []
    for content in combined_data["pdf_data"].values():
        chunks.extend([chunk.strip() for chunk in content.split(".") if chunk.strip()])

    print(f"전체 청크 수: {len(chunks)}")
    

    # FAISS 인덱스 확인 및 로드
    index = check_faiss_index(FAISS_INDEX)
    if not index:
        print("프로그램을 종료합니다. FAISS 인덱스를 생성하거나 경로를 확인하세요.")
        return
    
    # 임베딩 초기화
    embeddings = UpstageEmbeddings(upstage_api_key=UPSTAGE_API_KEY, model="solar-embedding-1-large-query")
    
    # FAISSRetrieverWithCohere 초기화
    retriever = initialize_retriever(FAISS_INDEX, chunks, UPSTAGE_API_KEY)
    print("Retriever 초기화 완료")

    # RAG 체인 
    rag_chain = create_rag_chain(chunks, retriever, OPENAI_API_KEY)
    print("RAG 체인 생성 완료")

    # 터미널에서 잘 돌아가는 지 확인 
    while True:
        query = input("\n저는 당신의 육아 도우미 챗봇입니다. 궁금한 점을 입력해주세요!")
        if query.lower() == "exit":
            print("추가적인 정보를 알고 싶다면 보건복지부 홈페이지에 방문해보세요.")
            break
        response = get_rag_response(rag_chain, query)
        print(f"답변: {response}")

if __name__ == "__main__":
    main()


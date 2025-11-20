import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# -------------------------------
# 1. Load PDF
# -------------------------------
loader = PyPDFLoader("sample.pdf")
pages = loader.load()

# -------------------------------
# 2. Split text
# -------------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
docs = splitter.split_documents(pages)

# -------------------------------
# 3. Embeddings + Chroma
# -------------------------------
embeddings = SentenceTransformerEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectordb = Chroma.from_documents(docs, embeddings)
retriever = vectordb.as_retriever()

# -------------------------------
# 4. Groq LLM
# -------------------------------
llm = ChatGroq(
    groq_api_key=api_key,
    model_name="llama-3.3-70b-versatile"
)

# -------------------------------
# 5. Manual RAG function
# -------------------------------
def rag_query(question):
    # Retrieve top matching docs
    docs = retriever.invoke(question)   # ← FIXED
    # Combine the docs into context text
    context = "\n\n".join([d.page_content for d in docs])
    prompt = f"""
Use ONLY the context below to answer the question.
CONTEXT:
{context}
QUESTION: {question}
ANSWER:
"""
    response = llm.invoke(prompt)
    return response.content

# -------------------------------
# 6. Test RAG
# -------------------------------
query = "Give me a summary of this PDF in 5 points."
answer = rag_query(query)

print("\n=== FINAL ANSWER ===")
print(answer)

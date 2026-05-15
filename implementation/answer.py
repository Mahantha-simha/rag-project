from pathlib import Path
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from dotenv import load_dotenv

load_dotenv(override=True)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
DB_NAME = str(Path(__file__).parent.parent / "vector_db")
RETRIEVAL_K = 10
SYSTEM_PROMPT = "You are a knowledgeable assistant for Insurellm.\nContext:\n{context}"

vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
retriever = vectorstore.as_retriever()
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)

def fetch_context(question):
    return retriever.invoke(question, k=RETRIEVAL_K)

def answer_question(question, history=[]):
    docs = fetch_context(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    messages = [SystemMessage(content=SYSTEM_PROMPT.format(context=context))]
    messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(content=question))
    return llm.invoke(messages).content, docs

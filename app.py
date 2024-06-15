import os 

from dotenv import load_dotenv
load_dotenv()

from langchain_community.embeddings import JinaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate 
from langchain.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from mistralai.client import MistralClient
from langchain_mistralai.chat_models import ChatMistralAI

jina_api_key=os.environ["jina_api_key"]
api_key = os.environ["MISTRAL_API_KEY"]

model="mistral-medium-latest"
llm = ChatMistralAI(api_key=api_key,model=model)

#llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=api_key, temperature=0.7)


instructor_embeddings=JinaEmbeddings(
jina_api_key=jina_api_key, model_name="jina-embeddings-v2-base-en"
)
vectordb_file_path='faiss_index'

def create_vector_db():
    file_path=vectordb_file_path
    
    #load the pdf 
    pdf_path='Maroc - Code travail.pdf'
    loader = PyMuPDFLoader(pdf_path)
    data = loader.load()

    # split the the pdf to chunks
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    chunks=text_splitter.split_documents(data)
    
    vectordb=FAISS.from_documents(documents=chunks,
                                 embedding=instructor_embeddings,
                                 )
    vectordb.save_local(vectordb_file_path)
    
    
def get_qa_chain():
    vectordb=FAISS.load_local(vectordb_file_path, instructor_embeddings, allow_dangerous_deserialization=True)
    
    #retriever:
    retriever = vectordb.as_retriever(score_threshold=0.7)
    
    prompt_template="""Étant donné le contexte et une question suivants, générez une réponse en vous basant uniquement sur ce contexte.
Dans la réponse, essayez de fournir autant de texte que possible provenant de la section "réponse" du document source, sans modifier le texte de manière significative.
Si la réponse ne se trouve pas dans le contexte, veuillez indiquer "Je ne sais pas". N'inventez pas de réponse.
    CONTEXT: {context}

    QUESTION: {question}"""
    
    PROMPT = PromptTemplate( template=prompt_template, input_variables=['context', 'question'])
    
    chain_type_kwargs= {'prompt': PROMPT}
    
    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type='stuff',
                                        retriever=retriever,
                                        input_key='query',
                                        return_source_documents=True,
                                        chain_type_kwargs=chain_type_kwargs)
    return chain

if __name__ == "__main__":
    create_vector_db()
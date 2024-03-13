import streamlit as st
import os
from dotenv import load_dotenv
from pypdf import PdfReader
import pickle
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import AzureChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

# convert the pdf to text chunks
def process_text(pdf, chuck_size, chuck_overlap):
    pdf_reader = PdfReader(pdf)
    # extract the text from the PDF
    page_text = ""
    for page in pdf_reader.pages:
        page_text += page.extract_text()
   
   #  print(page_text)
    # split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chuck_size,
        chunk_overlap=chuck_overlap,
        length_function=len
        )
    chunks = text_splitter.split_text(text=page_text)
    print(page_text)
    if chunks:
        return chunks
    else:
        raise ValueError("Could not process text in PDF")

def get_embeddings(chunks, pdf_path):
    embeddings = AzureOpenAIEmbeddings()
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    if vector_store is not None:
        return vector_store
    else:
        raise ValueError("Issue creating and saving vector store")

    
@st.cache_resource
def load_data(vector_store_dir: str = "data/amazon-food-reviews-faiss"):
    pdf_path = '/Users/i547603/githubRepo/streamlit-amazon-food-review/IRM Help.pdf'
    chuck_size = 500
    chuck_overlap = 100

    chunks = process_text(pdf_path, chuck_size, chuck_overlap)
    vector_store = get_embeddings(chunks, pdf_path)

    llm = AzureChatOpenAI(model_name="gpt-35-turbo", temperature=0.5)

    print("Loading data...")

    AMAZON_REVIEW_BOT = RetrievalQA.from_chain_type(
        llm,
        retriever=vector_store.as_retriever(search_type="similarity_score_threshold",
                                            search_kwargs={"score_threshold": 0.7})
    )
    AMAZON_REVIEW_BOT.return_source_documents = True

    return AMAZON_REVIEW_BOT



def chat(message, history):
    print(f"[message]{message}")
    print(f"[history]{history}")
    enable_chat = True

    AMAZON_REVIEW_BOT = load_data()

    ans = AMAZON_REVIEW_BOT.invoke({"query": message})
    if ans["source_documents"] or enable_chat:
        return ans["result"]
    else:
        return "I don't know."

if __name__ == "__main__":
    os.environ["OPENAI_API_TYPE"] = "azure"
    os.environ["OPENAI_API_VERSION"] = "2023-05-15"
    os.environ["OPENAI_API_BASE"] = "https://pvg-azure-openai-uk-south.openai.azure.com/openai"
    env_path = os.getenv("HOME") + "/Documents/src/openai/.env"
    load_dotenv(dotenv_path=env_path, verbose=True) 

    st.title('IRM Help')

    prompt = st.chat_input("Enter your questions here")

    if "user_prompt_history" not in st.session_state:
       st.session_state["user_prompt_history"]=[]
    if "chat_answers_history" not in st.session_state:
       st.session_state["chat_answers_history"]=[]
    if "chat_history" not in st.session_state:
       st.session_state["chat_history"]=[]

    if prompt:
       with st.spinner("Generating......"):
           output = chat(prompt, st.session_state["chat_history"])

           st.session_state["chat_answers_history"].append(output)
           st.session_state["user_prompt_history"].append(prompt)
           st.session_state["chat_history"].append((prompt,output))

    # Displaying the chat history

    if st.session_state["chat_answers_history"]:
       for i, j in zip(st.session_state["chat_answers_history"],st.session_state["user_prompt_history"]):
          message1 = st.chat_message("user")
          message1.write(j)
          message2 = st.chat_message("assistant")
          message2.write(i)

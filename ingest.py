"""Load html from files, clean up, split, ingest into Weaviate."""
import pickle

from langchain.document_loaders import ReadTheDocsLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
from langchain.chains.llm import LLMChain
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ChatVectorDBChain
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains.chat_vector_db.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import IFixitLoader

def ingest_docs():
    """Get documents from web pages."""
    '''
    loader = ReadTheDocsLoader("langchain.readthedocs.io/en/latest/")
    raw_documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    documents = text_splitter.split_documents(raw_documents)
    embeddings = OpenAIEmbeddings()
    '''
    urls = ["https://www.ifixit.com/Device/Xbox_One",
            "https://www.ifixit.com/Answers/View/192889/Xbox+Powers+on+and+immediately+powers+off",
            "https://www.ifixit.com/Device/Xbox",
            "https://www.ifixit.com/Device/Xbox_360",
            "https://www.ifixit.com/Device/Xbox_360_S",
            "https://www.ifixit.com/Device/Xbox_One",
            "https://www.ifixit.com/Device/Microsoft_Game_Console_Accessory",
            "https://www.ifixit.com/Device/Xbox_360_E",
            "https://www.ifixit.com/Device/Xbox_One_S",
            "https://www.ifixit.com/Device/Xbox_One_X",
            "https://www.ifixit.com/Device/Xbox_One_X_Project_Scorpio_Edition",
            "https://www.ifixit.com/Device/Xbox_One_S_All_Digital_Edition",
            "https://www.ifixit.com/Device/Xbox_Series_S",
            "https://www.ifixit.com/Device/Xbox_Series_X",
            "https://www.ifixit.com/Device/Xbox_360_Wireless_Controller",
            "https://www.ifixit.com/Device/Xbox_Series_X_Wireless_Controller",
            "https://www.ifixit.com/Device/Xbox_Adaptive_Controller",
            "https://www.ifixit.com/Device/Xbox_Adaptive_Controller",
            "https://www.ifixit.com/Device/Microsoft_Kinect",
            "https://www.ifixit.com/Device/Xbox_One_Wireless_Controller_Model_1537",
            "https://www.ifixit.com/Device/Xbox_One_Wireless_Controller_1697",
            "https://www.ifixit.com/Device/Xbox_One_Wireless_Controller_%28Model_1914%29"]

    corpus = []
    for url in urls:
        documents = IFixitLoader(url).load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        documents = text_splitter.split_documents(documents)
        corpus += documents
    from pprint import pprint
    pprint(corpus)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(corpus, embeddings)

    # Save vectorstore
    with open("vectorstore_ifixit.pkl", "wb") as f:
        pickle.dump(vectorstore, f)


if __name__ == "__main__":
    ingest_docs()

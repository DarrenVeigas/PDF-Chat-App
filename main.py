import streamlit as st
#from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter as RCTS
from langchain_community.embeddings.openai import OpenAIEmbeddings
import pickle,os
from langchain_community.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
#import tensorflow as tf
import tensorflow_hub as hub
import numpy
embed = hub.load("https://www.kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/universal-sentence-encoder/2")

def get_embeddings(texts):
  return embed(texts).numpy()
with st.sidebar:
  st.title('PDF-chat-app')
  st.markdown('''
  # About
  This app is an LLM powered chatbot built using 
  -[Streamlit](https://streamlit.io/)
  -[Langchain](https://python.langchain.com/)
  -[OpenAI](https://openai.com/)
  ''')
  #add_vertical_space(5)
  st.write('Made by Darren Veigas')

def main():
  st.header('Chat with PDF')
  pdf=st.file_uploader("Upload PDF",type='pdf')
  if pdf is not None:
    pdf_reader=PdfReader(pdf)
    text=''
    for page in pdf_reader.pages:
      text+=page.extract_text()
    text_splitter=RCTS(
      chunk_size=1000,
      chunk_overlap=200,
      length_function=len,
    )  
    chunks=text_splitter.split_text(text=text)

    store_name=pdf.name[:-4]
    if os.path.exists(f"{store_name}.pkl"):
      with open(f"{store_name}.pkl",'rb') as f:
        VectorStore=pickle.load(f)
    else:      
      embeddings=OpenAIEmbeddings()
      VectorStore=FAISS.from_texts(chunks,embeddings)
      with open(f"{store_name}.pkl",'wb') as f:
        pickle.dump(embeddings,f)
    
    query=st.text_input('Ask questions about your pdf file:')
    if query:
      docs=VectorStore.similarity_search(query=query,k=3)
      llm=OpenAI(temperature=0)
      chain=load_qa_chain(llm=llm,chain_type="stuff")
      response=chain.run(input_documens=docs,question=query)
      st.write(response)





if __name__=='__main__':
  main()
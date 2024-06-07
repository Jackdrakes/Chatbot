from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.embeddings import HuggingFaceEmbeddings

loader = PyPDFDirectoryLoader("pdfs")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)

groq = "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"

embeddings = HuggingFaceEmbeddings(
    model_name="mixedbread-ai/mxbai-embed-large-v1",
    encode_kwargs = {'precision': 'binary'}
    )
one_bit_vectorstore = FAISS.from_documents(documents, embeddings)
one_bit_retriever = one_bit_vectorstore.as_retriever(search_kwargs={"k": 3})

prompt = ChatPromptTemplate.from_template("""
your name is schemebot
you are a helpful chatbot for peoples.
Answer the following question based only on the provided context:

<context>
{context}
</context>

Don't give the reponse starts with according to the provided context.

Question: {input}
""")
model = ChatGroq(groq_api_key=groq, model_name="mixtral-8x7b-32768")

document_chain = create_stuff_documents_chain(model, prompt)
one_bit_retrieval_chain = create_retrieval_chain(one_bit_retriever, document_chain)

response = one_bit_retrieval_chain.invoke({"input": "what's your name"})
print(response["answer"])
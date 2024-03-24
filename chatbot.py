from fastapi import FastAPI
import uvicorn
from torch import cuda, bfloat16
import torch
import transformers
from transformers import AutoTokenizer
from time import time
#import chromadb
#from chromadb.config import Settings
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


access_token = ""
model_id="meta-llama/Llama-2-7b-chat-hf"

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
print("*"*50)
print(device)
print("*"*50)


bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)


time_1 = time()
model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    use_auth_token=access_token,
)


model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto',
    use_auth_token=access_token,
)

tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=access_token,)
time_2 = time()
print(f"Prepare model, tokenizer: {round(time_2-time_1, 3)} sec.")


time_1 = time()
query_pipeline = transformers.pipeline(
    "text-generation",
    model = model,
    tokenizer = tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
)
time_2 = time()
print(f"Prepare pipeline: {round(time_2-time_1,3)} sec.")


def test_model(tokenizer, pipeline, prompt_to_test):
    time_1 = time()
    sequences = pipeline(

        prompt_to_test,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=200,
    )

    time_2 = time()
    print(f"Test inference: {round(time_2-time_1, 3)} sec.")
    for seq in sequences:
        print(f"Result: {seq['generated_text']}")



# test_model(tokenizer,
#            query_pipeline,
#            "Do you like me?")
llm = HuggingFacePipeline(pipeline=query_pipeline)

# llm(prompt="Please explain where Poland is located. ")

loader = TextLoader("data/attendees_guide_cloudfest.txt",
                    encoding="utf8")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
all_splits = text_splitter.split_documents(documents)

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cuda"}

embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
vectordb = Chroma.from_documents(documents=all_splits, embedding=embeddings, persist_directory="chroma_db")
retriever = vectordb.as_retriever()

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    verbose=True
)

def test_rag(qa, query):
    print(f"Query: {query}\n")
    time_1 = time()
    result = qa.run(query)
    time_2 = time()
    print(f"Inference time: {round(time_2-time_1, 3)} sec.")
    print("\nResult: ", result)
    return result


app = FastAPI()

origins = [
        "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_source(query):
    docs = vectordb.similarity_search(query)
    for doc in docs:
        doc_details = doc.to_json()['kwargs']
        source = doc_details['metadata']['source']
        text = doc_details['page_content']
        # print("Source: ", doc_details['metadata']['source'])
        # print("Text: ", doc_details['page_content'], "\n")
    return source, text

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/question")
def rag_question(query: str):
    try:
        result = test_rag(qa, query)
        source, text = get_source(query)
        return {"query": query, "answer": result, "source": source, "text":text,}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)

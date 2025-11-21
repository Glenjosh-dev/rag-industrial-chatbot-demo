import argparse
import gradio as gr
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

def load_vectorstore(index_path: str):
    embeddings = OpenAIEmbeddings()
    vectordb = FAISS.load_local(index_path, embeddings)
    return vectordb

def main(index_path: str, model="gpt-4o-mini"):
    vectordb = load_vectorstore(index_path)
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    llm = ChatOpenAI(model=model, temperature=0)

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    with gr.Blocks() as demo:
        gr.Markdown("# RAG Industrial Chatbot Demo")
        chat = gr.Chatbot()
        msg = gr.Textbox(placeholder="Ask something about the documents...")
        state = gr.State([])

        def respond(message, history):
            result = qa({"question": message, "chat_history": history})
            answer = result["answer"]
            history = history + [(message, answer)]
            return "", history

        msg.submit(respond, [msg, state], [msg, state])
        demo.launch()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_path", default="./faiss_index/faiss_index")
    args = parser.parse_args()
    main(args.index_path)

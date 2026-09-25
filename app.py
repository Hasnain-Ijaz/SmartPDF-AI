import streamlit as st
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

st.set_page_config(page_title="PDF RAG Assistant", page_icon="📄", layout="wide")
st.title("📄 PDF RAG Assistant")
st.caption("Upload a PDF, then ask questions grounded in its content | Powered by Groq + FAISS | Presented by Hasnain")

def get_groq_api_key():
    try:
        if hasattr(st, 'secrets') and 'GROQ_API_KEY' in st.secrets:
            return st.secrets['GROQ_API_KEY']
    except Exception:
        pass
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY not found. Add it in Streamlit Cloud Secrets.")
        st.stop()
    return api_key

GROQ_API_KEY = get_groq_api_key()

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    process_button = st.button("Process Document", type="primary", use_container_width=True)
    if st.session_state.pdf_processed:
        st.success("Document indexed - ready for questions")
    st.divider()
    if st.button("Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if uploaded_file and process_button and not st.session_state.pdf_processed:
    temp_path = "temp_uploaded.pdf"
    try:
        with st.status("Processing PDF...", expanded=True) as status:
            st.write("Saving file...")
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.write("Extracting PDF text...")
            loader = PyPDFLoader(temp_path)
            documents = loader.load()

            if not documents:
                st.error("No text extracted. PDF might be scanned images.")
                st.stop()

            st.write("Splitting into chunks...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
            )
            chunks = text_splitter.split_documents(documents)

            if not chunks:
                st.error("No chunks created. Try a different PDF.")
                st.stop()

            st.write(f"Created {len(chunks)} chunks")

            st.write("Generating embeddings...")
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )

            st.write("Storing in FAISS...")
            vectorstore = FAISS.from_documents(chunks, embeddings)

            st.write("Building QA chain...")
            llm = ChatGroq(
                groq_api_key=GROQ_API_KEY,
                model_name="openai/gpt-oss-120b",
                temperature=0.1
            )

            # Strict prompt: sirf context se jawab do, warna specific message do
            prompt_template = """You are a PDF question-answering assistant.

STRICT RULES:
1. Answer ONLY using the information provided in the context below.
2. If the answer is NOT present in the context, respond with EXACTLY this sentence:
   "This information is not present in the uploaded PDF."
3. Do NOT use any outside knowledge.
4. Do NOT make up answers.
5. If the context is partially relevant, answer only what is supported.

Context:
{context}

Question: {question}

Answer:"""

            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )

            # Threshold hata di — sirf top-k chunks return karo
            # (Prompts hi LLM ko strict rakhenge)
            retriever = vectorstore.as_retriever(
                search_kwargs={"k": 4}
            )

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT}
            )
            st.session_state.qa_chain = qa_chain
            st.session_state.pdf_processed = True
            status.update(label="Processing complete!", state="complete", expanded=False)

    except Exception as e:
        st.error(f"Processing failed: {str(e)}")
        st.stop()
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if st.session_state.pdf_processed and st.session_state.qa_chain:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if question := st.chat_input("Ask a question about the document..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.qa_chain.invoke({"query": question})
                    answer = result["result"]
                    st.markdown(answer)

                    # Sources sirf tab dikhao jab answer PDF se aaya ho
                    not_found_message = "This information is not present in the uploaded PDF."
                    has_answer = not_found_message.lower() not in answer.lower()

                    if not has_answer:
                        st.info("ℹ️ Ye jawab PDF mein nahi mila. Aap koi doosra sawal pooch sakte hain.")
                    elif result.get("source_documents"):
                        with st.expander("View sources"):
                            for i, doc in enumerate(result["source_documents"], 1):
                                page = doc.metadata.get("page", "unknown")
                                page_display = page + 1 if isinstance(page, int) else page
                                st.markdown(f"Chunk {i} (page {page_display})")
                                st.text(doc.page_content[:500])

                    st.session_state.messages.append({"role": "assistant", "content": answer})

                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

elif not st.session_state.pdf_processed:
    st.info("Upload a PDF and click 'Process Document' in the sidebar.")
    with st.expander("How to use"):
        st.markdown(
            "1. Upload a PDF and click Process Document\n\n"
            "2. Ask questions in the chat input"
        )

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import os

def get_vector_store():

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    if not os.path.exists("./chroma_db"):
        all_docs = []
        # code to create chromadib document store
        print(f"loading folders")
        folders = [
            "knowledge_files/aventro",
            "knowledge_files/cloudway/pdf"
        ]

        # extension map
        loader_map = {
            ".pdf": PyPDFLoader,
            ".md": UnstructuredMarkdownLoader
        }
        print(f"building store")
        for folder in folders:
            for ext, load_class in loader_map.items():
                dir_loader = DirectoryLoader(
                    folder,
                    glob = f"**/*{ext}",
                    loader_cls=load_class,
                    show_progress=True,
                    use_multithreading=True
                )
                docs = dir_loader.load()

                # mark metadata
                for doc in docs:
                    doc.metadata['source_folder'] = os.path.basename(os.path.dirname(folder)) # get second last
                all_docs.extend(docs) # take all the docs from a source folder and add it into all_docs
        print(f"loaded all {len(all_docs)} documents")

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(all_docs)

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="./chroma_db",
            collection_name="my_knowledge_base"
        )
        print(f"vectorstore built and persisted")
    else:
        # vs = Chroma(
        #     persist_directory="./chroma_db",
        #     embedding_function=embeddings,
        #     collection_name="my_knowledge_base"
        # )
        print(f"Vector Store already exists")

if __name__ == "__main__":
    get_vector_store()
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import Docx2txtLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub

# Загружаем документ
loader = Docx2txtLoader('user_guide.docx')
docs = loader.load()

# Разбиваем документ на части
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Переводим части документа в эмбединги и помещаем в хранилище
embedding = HuggingFaceEmbeddings(
	model_name='sentence-transformers/distiluse-base-multilingual-cased',
	model_kwargs={'device':'cuda:0'},
	encode_kwargs={'normalize_embeddings': False}
)
vectorstore = Chroma.from_documents(documents=splits, embedding=embedding)
retriever = vectorstore.as_retriever()

# Подгружаем LLM с HF
llm = HuggingFaceHub(repo_id='lmsys/vicuna-13b-v1.5-16k')

# Формируем шаблон запроса к llm
template = """Используй следующие фрагменты контекста, чтобы в конце ответить на вопрос.
Если ты не нашел ответа, просто скажи, что не знаешь ответа. Не пытайся выдумывать ответ.
Используй максимум три предложения и старайся отвечать максимально кратко.
{context}
Вопрос: {question}
Полезный ответ: """
prompt = PromptTemplate.from_template(template)

def format_docs(docs):
	return "\n\n".join(doc.page_content for doc in docs)

# Формируем конвеер
rag_chain = (
	{"context": retriever | format_docs, "question": RunnablePassthrough()}
	| prompt
	| llm
	| StrOutputParser()
)

# Задаем вопрос по документу
out = rag_chain.invoke("Какое назначение у ЛИС?")

print(out)

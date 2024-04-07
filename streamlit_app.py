import os
from langchain.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Weaviate
import weaviate
from weaviate.embedded import EmbeddedOptions
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter

import streamlit as st
from langchain_community.llms import Ollama

st.set_page_config(page_title="Pesquisa por Fundamento Jurídico", page_icon="⚖️")

st.markdown("<h1 style='font-size:38px;'>⚖️ Pesquisa por Fundamento Jurídico</h1>", unsafe_allow_html=True)
# escrever no sidebar
st.sidebar.markdown('## 📄 Sobre a aplicação:')
# escrever no sidebar com letras menores
st.sidebar.caption('É uma ferramenta que tem o intuito de facilitar a busca por artigos e leis para fundamentar peças jurídicas')

# colocar divisoria no sidebar
st.sidebar.markdown('---')

openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')

def inicializa(openai_api_key):
    # Caminho da pasta no Google Drive onde estão os arquivos
    caminho_da_pasta = 'trab_deep/docs'

    # Lista para armazenar os documentos
    documents_list = []

    # Percorrer os diretórios e carregar os arquivos
    for root, dirs, files in os.walk(caminho_da_pasta):
        for nome_arquivo in files:
            if nome_arquivo.endswith('.txt'):  # Verificar se o arquivo é um arquivo de texto
                caminho_completo = os.path.join(root, nome_arquivo)
                loader = TextLoader(caminho_completo, autodetect_encoding=True)
                documents = loader.load()
                
                # Adicionar o documento à lista de documentos
                documents_list.extend(documents)

    # Agora, documents_list contém todos os documentos carregados

    #Chunks de 500 palavras
    text_splitter = CharacterTextSplitter(chunk_size=500)
    chunks = text_splitter.split_documents(documents_list)

    client = weaviate.Client(
        embedded_options = EmbeddedOptions()
    )

    vectorstore = Weaviate.from_documents(
        client = client,
        documents = chunks[0:20],
        embedding = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key),  #text-embedding-3-small usado para criar os embeddings
        by_text = False
    )

    retriever = vectorstore.as_retriever()

    template = """Você é um assistente que irá retornar artigos e leis que podem ser aplicados no contexto fornecido.
    Sempre especifique o nome da lei ou do estatuto em que os artigos se encontram.
    Forneça explicação geral dos artigos e leis, considerando o contexto.
    Não invente.
    Se não houver pergunta direta, responda as leis e artigos que podem ser aplicados no contexto fornecido.
    Use as seguintes peças de texto para responder a pergunta.
    Se não souber a resposta, apenas responda que não sabe a resposta.
    Use no máximo sete sentenças e mantenha a resposta concisa.
    Pergunta: {question}
    Contexto: {context}
    Resposta:/
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    return prompt, retriever

def generate_response(input_text, prompt, retriever, openai_api_key):
    #print(prompt)
    
    
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0, openai_api_key=openai_api_key)

    rag_chain = (
        {"context": retriever,  "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    #query = "Contexto: Uma criança estava sofrendo maus tratos dos pais, quais artigos eu poderia utilizar para defender a criança?"
    result = rag_chain.invoke(input_text)
    return result


with st.form('my_form'):
    text = st.text_area('Digite o contexto:', key='Para qual contexto você precisa de fundamento jurídico?', help='Para qual contexto você precisa de fundamento jurídico?')
    
    submitted = st.form_submit_button('Enviar')
    if not openai_api_key.startswith('sk-'):
        st.warning('Por favor, entre com sua OpenAi API key!', icon='⚠')
        
    if submitted and openai_api_key.startswith('sk-'):
        global prompt_global, retriever_global
    
        if 'prompt_global' not in globals():
            prompt_global, retriever_global = inicializa(openai_api_key)
        resultado = generate_response(text, prompt_global, retriever_global, openai_api_key)
        # imprimir o resultado
        st.caption(f"**Resposta:** {resultado}")
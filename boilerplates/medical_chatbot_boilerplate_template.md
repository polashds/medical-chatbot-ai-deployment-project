# create git repo
# create template.sh/
# creating directories for a project structure
mkdir -p src
mkdir -p research

# cerating files for the project
touch src/__init__.py
touch src/helpers.py
touch src/prompt.py
touch .env
touch setup.py
touch app.py
touch research/trials.ipynb
touch requirements.txt
touch README.md

echo "Project structure created successfully."

>sh template
>git status
>git add .
>git commit -m "Initial commit"
>git commit origin mian
>pip list
>python --version
>conda env list
>conda create -n medibot python=3.10 -y
>conda activate medibot

# setup.py
from setuptools import find_packages, setup

setup(
    name="medical_chatbot",
    version="0.1.0",
    author="Boktiar Ahmed Bappy",
    author_email="entbappy73@gmail.com",
    packages=find_packages(),
    install_requires=[]
)

# requirements.txt
langchain==0.3.26
flask==3.1.1 
sentence-transformers==4.1.0
pypdf==5.6.1 
python-dotenv==1.1.0
langchain-pinecone==0.2.8 
langchain-openai==0.3.24
langchain-community==0.3.26
-e .

>pip install -r requirements.txt
>

# experiment trials notebook
google search for the sentence transformer/all-MiniLM-L6-v2
-download required model from huggingface model hub

# modular coding
collect code from notebook experiment in helper.py
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document


#Extract Data From the PDF File
def load_pdf_file(data):
    loader= DirectoryLoader(data,
                            glob="*.pdf",
                            loader_cls=PyPDFLoader)

    documents=loader.load()

    return documents



def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Given a list of Document objects, return a new list of Document objects
    containing only 'source' in metadata and the original page_content.
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return minimal_docs



#Split the Data into Text Chunks
def text_split(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks



# Download the Embeddings from HuggingFace 
def download_hugging_face_embeddings():
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')  #this model return 384 dimensions
    return embeddings


# collect code from notebook research in prompt.py
system_prompt = (
    "You are an Medical assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)


for storing the vector embeddings of the input text, create a new file called embeddings.py/store_index.py
# Store the embeddings in a file
 before running  embeddings.py/store_index.py delete the existing index file in pinecone webiste to create again
 from dotenv import load_dotenv
import os
from src.helper import load_pdf_file, filter_to_minimal_docs, text_split, download_hugging_face_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec 
from langchain_pinecone import PineconeVectorStore

load_dotenv()


PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


extracted_data=load_pdf_file(data='data/')
filter_data = filter_to_minimal_docs(extracted_data)
text_chunks=text_split(filter_data)

embeddings = download_hugging_face_embeddings()

pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)



index_name = "medical-chatbot"  # change if desired

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)


docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings, 
)
 >python store_index.py

 google search free image embedding models

 # write code in app.py
 from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os


app = Flask(__name__)


load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot" 
# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)




retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

chatModel = ChatOpenAI(model="gpt-4o")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)



@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)

# create templates folder and add chat.html to it
google search free chatbot templates 
<link href="//maxcdn.bootstrapcdn.com/bootstrap/4.1.1/css/bootstrap.min.css" rel="stylesheet" id="bootstrap-css">
<script src="//maxcdn.bootstrapcdn.com/bootstrap/4.1.1/js/bootstrap.min.js"></script>
<script src="//cdnjs.cloudflare.com/ajax/libs/jquery/3.2.1/jquery.min.js"></script>

<!DOCTYPE html>
<html>
	<head>
		<title>Chatbot</title>
		<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css" integrity="sha384-MCw98/SFnGE8fJT3GXwEOngsV7Zt27NXFoaoApmYm81iuXoPkFOJwJ8ERdknLPMO" crossorigin="anonymous">
		<link rel="stylesheet" href="https://use.fontawesome.com/releases/v5.5.0/css/all.css" integrity="sha384-B4dIYHKNBt8Bc12p+WXckhzcICo0wtJAoU8YZTY5qE0Id1GSseTk6S+L3BlXeVIU" crossorigin="anonymous">
		<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js"></script>
		<link rel="stylesheet" type="text/css" href="{{ url_for('static', filename='style.css')}}"/>
	</head>
	
	
	<body>
		<div class="container-fluid h-100">
			<div class="row justify-content-center h-100">		
				<div class="col-md-8 col-xl-6 chat">
					<div class="card">
						<div class="card-header msg_head">
							<div class="d-flex bd-highlight">
								<div class="img_cont">
									<img src="https://cdn-icons-png.flaticon.com/512/387/387569.png" class="rounded-circle user_img">
									<!-- <img src="https://www.prdistribution.com/spirit/uploads/pressreleases/2019/newsreleases/d83341deb75c4c4f6b113f27b1e42cd8-chatbot-florence-already-helps-thousands-of-patients-to-remember-their-medication.png" class="rounded-circle user_img"> -->
									<span class="online_icon"></span>
								</div>
								<div class="user_info">
									<span>Medical Chatbot</span>
									<p>Ask me anything!</p>
								</div>
							</div>
						</div>
						<div id="messageFormeight" class="card-body msg_card_body">
							
							
						</div>
						<div class="card-footer">
							<form id="messageArea" class="input-group">
                                <input type="text" id="text" name="msg" placeholder="Type your message..." autocomplete="off" class="form-control type_msg" required/>
								<div class="input-group-append">
									<button type="submit" id="send" class="input-group-text send_btn"><i class="fas fa-location-arrow"></i></button>
								</div>
							</form>
						</div>
					</div>
				</div>
			</div>
		</div>
		
		<script>
			$(document).ready(function() {
				$("#messageArea").on("submit", function(event) {
					const date = new Date();
					const hour = date.getHours();
					const minute = date.getMinutes();
					const str_time = hour+":"+minute;
					var rawText = $("#text").val();

					var userHtml = '<div class="d-flex justify-content-end mb-4"><div class="msg_cotainer_send">' + rawText + '<span class="msg_time_send">'+ str_time + '</span></div><div class="img_cont_msg"><img src="https://i.ibb.co/d5b84Xw/Untitled-design.png" class="rounded-circle user_img_msg"></div></div>';
					
					$("#text").val("");
					$("#messageFormeight").append(userHtml);

					$.ajax({
						data: {
							msg: rawText,	
						},
						type: "POST",
						url: "/get",
					}).done(function(data) {
						var botHtml = '<div class="d-flex justify-content-start mb-4"><div class="img_cont_msg"><img src="https://cdn-icons-png.flaticon.com/512/387/387569.png" class="rounded-circle user_img_msg"></div><div class="msg_cotainer">' + data + '<span class="msg_time">' + str_time + '</span></div></div>';
						$("#messageFormeight").append($.parseHTML(botHtml));
					});
					event.preventDefault();
				});
			});
		</script>
        
    </body>
</html>

# create static folder and add styl.css file

body,html{
	height: 100%;
	margin: 0;
	background: rgb(44, 47, 59);
   background: -webkit-linear-gradient(to right, rgb(40, 59, 34), rgb(54, 60, 70), rgb(32, 32, 43));
	background: linear-gradient(to right, rgb(38, 51, 61), rgb(50, 55, 65), rgb(33, 33, 78));
}

.chat{
	margin-top: auto;
	margin-bottom: auto;
}
.card{
	height: 500px;
	border-radius: 15px !important;
	background-color: rgba(0,0,0,0.4) !important;
}
.contacts_body{
	padding:  0.75rem 0 !important;
	overflow-y: auto;
	white-space: nowrap;
}
.msg_card_body{
	overflow-y: auto;
}
.card-header{
	border-radius: 15px 15px 0 0 !important;
	border-bottom: 0 !important;
}
.card-footer{
border-radius: 0 0 15px 15px !important;
	border-top: 0 !important;
}
.container{
	align-content: center;
}
.search{
	border-radius: 15px 0 0 15px !important;
	background-color: rgba(0,0,0,0.3) !important;
	border:0 !important;
	color:white !important;
}
.search:focus{
	 box-shadow:none !important;
   outline:0px !important;
}
.type_msg{
	background-color: rgba(0,0,0,0.3) !important;
	border:0 !important;
	color:white !important;
	height: 60px !important;
	overflow-y: auto;
}
	.type_msg:focus{
	 box-shadow:none !important;
   outline:0px !important;
}
.attach_btn{
	border-radius: 15px 0 0 15px !important;
	background-color: rgba(0,0,0,0.3) !important;
	border:0 !important;
	color: white !important;
	cursor: pointer;
}
.send_btn{
	border-radius: 0 15px 15px 0 !important;
	background-color: rgba(0,0,0,0.3) !important;
	border:0 !important;
	color: white !important;
	cursor: pointer;
}
.search_btn{
	border-radius: 0 15px 15px 0 !important;
	background-color: rgba(0,0,0,0.3) !important;
	border:0 !important;
	color: white !important;
	cursor: pointer;
}
.contacts{
	list-style: none;
	padding: 0;
}
.contacts li{
	width: 100% !important;
	padding: 5px 10px;
	margin-bottom: 15px !important;
}
.active{
	background-color: rgba(0,0,0,0.3);
}
.user_img{
	height: 70px;
	width: 70px;
	border:1.5px solid #f5f6fa;

}
.user_img_msg{
	height: 40px;
	width: 40px;
	border:1.5px solid #f5f6fa;

}
.img_cont{
	position: relative;
	height: 70px;
	width: 70px;
}
.img_cont_msg{
	height: 40px;
	width: 40px;
}
.online_icon{
	position: absolute;
	height: 15px;
	width:15px;
	background-color: #4cd137;
	border-radius: 50%;
	bottom: 0.2em;
	right: 0.4em;
	border:1.5px solid white;
}
.offline{
	background-color: #c23616 !important;
}
.user_info{
	margin-top: auto;
	margin-bottom: auto;
	margin-left: 15px;
}
.user_info span{
	font-size: 20px;
	color: white;
}
.user_info p{
	font-size: 10px;
	color: rgba(255,255,255,0.6);
}
.video_cam{
	margin-left: 50px;
	margin-top: 5px;
}
.video_cam span{
	color: white;
	font-size: 20px;
	cursor: pointer;
	margin-right: 20px;
}
.msg_cotainer{
	margin-top: auto;
	margin-bottom: auto;
	margin-left: 10px;
	border-radius: 25px;
	background-color: rgb(82, 172, 255);
	padding: 10px;
	position: relative;
}
.msg_cotainer_send{
	margin-top: auto;
	margin-bottom: auto;
	margin-right: 10px;
	border-radius: 25px;
	background-color: #58cc71;
	padding: 10px;
	position: relative;
}
.msg_time{
	position: absolute;
	left: 0;
	bottom: -15px;
	color: rgba(255,255,255,0.5);
	font-size: 10px;
}
.msg_time_send{
	position: absolute;
	right:0;
	bottom: -15px;
	color: rgba(255,255,255,0.5);
	font-size: 10px;
}
.msg_head{
	position: relative;
}
#action_menu_btn{
	position: absolute;
	right: 10px;
	top: 10px;
	color: white;
	cursor: pointer;
	font-size: 20px;
}
.action_menu{
	z-index: 1;
	position: absolute;
	padding: 15px 0;
	background-color: rgba(0,0,0,0.5);
	color: white;
	border-radius: 15px;
	top: 30px;
	right: 15px;
	display: none;
}
.action_menu ul{
	list-style: none;
	padding: 0;
	margin: 0;
}
.action_menu ul li{
	width: 100%;
	padding: 10px 15px;
	margin-bottom: 5px;
}
.action_menu ul li i{
	padding-right: 10px;
}
.action_menu ul li:hover{
	cursor: pointer;
	background-color: rgba(0,0,0,0.2);
}
@media(max-width: 576px){
	.contacts_card{
	margin-bottom: 15px !important;
}
}

>python app.py

to add memory google search converstion buffer  memory or from dswithbappy how to add memory to chatbot
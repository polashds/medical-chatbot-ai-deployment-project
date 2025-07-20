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
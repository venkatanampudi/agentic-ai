# ****************Deploying the application using LangServe****************

# Install the necessary packages
# pip install fastapi
# pip install uvicorn
# pip install langserve
# pip install sse_starlette (implement the stream and stream_log endpoints)
# update the requirements.txt file accordingly

# import the necessary keys from ".env" file
import os 
from dotenv import load_dotenv
load_dotenv()
groq_api_key=os.getenv("GROQ_API_KEY")

# import the necessary libraries
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langserve import add_routes  # add_routes help you to create APIs


#1.Create a model 
model=ChatGroq(model="groq/compound",groq_api_key=groq_api_key)

#2.Create a prompt template
system_template="Translate the following into {language}"
prompt_template=ChatPromptTemplate.from_messages(
    [
    ('system',system_template),
    ('user','{text}')
     ]
)

#3.Create a parser
parser=StrOutputParser()

#4.Create a chain
chain=prompt_template|model|parser

#5. App definition 
app=FastAPI(title="Langchain",version="1.0",description="A simplel API using Langchain runnable interfaces")

#6.Adding chain routes
add_routes(
    app,
    chain,
    path="/chain"
)

#7.Calling the serve.py program
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost",port=8000)

#after successful execution, you must see langserve Swagger UI with APIs defined in this application.
# you can access the application at http://localhost:8000/docs 
# you can also test the same APIs in postman also with the same http://localhost:8000/ and given JSON body

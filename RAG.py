import concurrent.futures
import hashlib
import logging
import os
from datetime import datetime
from typing import List,Dict,Optional,Tuple,Union

import pandas as pd
from dotenv import load_dotenv
from langchain.output_parsers import StructuredOutputParser,ResponseSchema
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from sqlalchemy import create_engine
from tqdm import tqdm
from langchain_community.chat_models import ChatTongyi

load_dotenv(override=True)

DEEPSEEK_API_KEY=os.getenv("DEEPSEEK_API_KEY")
KIMI_API_KEY=os.getenv("KIMI_API_KEY")
GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
ALIYUN_API_KEY=os.getenv("DASHSCOPE_API_KEY")


model=ChatTongyi(api_key=ALIYUN_API_KEY)

question="王云琦是不是pig?"

thing=model.invoke(f"问题{question}中提到的事物是什么？只需要回答事物的中文名称。").content
man=model.invoke(f"问题{question}中提到的人是谁？只需要给出人名。").content
answer_prompt=f"只能回答{man}是或yes或不如{thing},并给出理由。"

result=model.invoke(f"{question}{answer_prompt}")
print(result.content)
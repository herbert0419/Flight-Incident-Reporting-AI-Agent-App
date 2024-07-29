import streamlit as st
import os
import base64
from pydub import AudioSegment
from openai import OpenAI
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_openai import AzureChatOpenAI
from docx import Document
from io import BytesIO
import base64


load_dotenv()

def generate_docx(result):
    doc = Document()
    doc.add_heading('Healthcare Diagnosis and Treatment Recommendations', 0)
    doc.add_paragraph(result)
    bio = BytesIO()
    doc.save(bio)
    bio.seek(0)
    return bio

def get_download_link(bio, filename):
    b64 = base64.b64encode(bio.read()).decode()
    return f'<a href="data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64,{b64}" download="{filename}">Download Flight Incident Report</a>'

# Initialize OpenAI client
groq = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1"
)

llm = AzureChatOpenAI(
    openai_api_version=os.environ["OPENAI_API_GPT_4_VERSION"],
    azure_deployment="gpt-4o",
    model="gpt-4o",
    temperature=0.7,
    openai_api_key=os.environ["OPENAI_API_GPT_4_KEY"],
    azure_endpoint=os.environ["OPENAI_API_GPT_4_BASE"]
)
# imports
import os

from dotenv import load_dotenv
import gradio as gr

# model and db initilaised
MODEL = 'gpt-4o-mini'
DB_NAME = 'vector-db'

#load environment cars
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
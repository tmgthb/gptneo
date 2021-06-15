import pandas as pd
import numpy as np
import streamlit as st
import torch
from transformers import pipeline, AutoModelForCausalLM
st.title('GPT-Neo')
st.text('This GPT-Neo model includes 125 Million parameters.')
prompt_text = st.text_input(label="Add here phrase, which you want to be completed", value="Add here few words")
generator = pipeline('text-generation', model='EleutherAI/gpt-neo-125M')
gpt_text = generator(prompt_text, do_sample=True, min_length=50)
st.text(gpt_text)

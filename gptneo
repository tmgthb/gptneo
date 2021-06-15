import pandas as pd
import numpy as np
import streamlit as st
import torch
from transformers import pipeline, AutoModelForCausalLM
st.title('GPT-Neo')
st.text('125 Million Parameter model')
prompt_text = st.text_input(label="Write few words", value="Insert you text here")
generator = pipeline('text-generation', model='EleutherAI/gpt-neo-125M')
gpt_text = generator(prompt_text, do_sample=True, min_length=50)
st.text(gpt_text)

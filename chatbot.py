# Install first (in terminal):
# pip install streamlit transformers torch

import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load pre-trained model
model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

st.set_page_config(page_title="AI Chatbot", page_icon="🤖")
st.title("🤖 Generative AI Chatbot")
st.write("Chat with an AI powered by Hugging Face DialoGPT!")

# Store conversation
if "chat_history_ids" not in st.session_state:
    st.session_state.chat_history_ids = None
if "past_inputs" not in st.session_state:
    st.session_state.past_inputs = []
if "past_responses" not in st.session_state:
    st.session_state.past_responses = []

# User input
user_input = st.text_input("You: ", key="input")

if st.button("Send") and user_input:
    # Encode new input
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    # Append to history
    if st.session_state.chat_history_ids is not None:
        bot_input_ids = torch.cat([st.session_state.chat_history_ids, new_input_ids], dim=-1)
    else:
        bot_input_ids = new_input_ids

    # Generate response
    st.session_state.chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode response
    bot_response = tokenizer.decode(
        st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0],
        skip_special_tokens=True
    )

    # Save history
    st.session_state.past_inputs.append(user_input)
    st.session_state.past_responses.append(bot_response)

# Display conversation
if st.session_state.past_responses:
    for i in range(len(st.session_state.past_responses)):
        st.write(f"🧑 You: {st.session_state.past_inputs[i]}")
        st.write(f"🤖 Bot: {st.session_state.past_responses[i]}")
        st.write("---")

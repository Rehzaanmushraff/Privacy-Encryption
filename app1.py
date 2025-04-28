import streamlit as st
import time
import base64
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
import json
import streamlit as st
import os
import pandas as pd
import base64
import subprocess
import time
import random
import string
from io import BytesIO
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
import hashlib
from gsheets import save_to_gsheet
import requests 
import qrcode
import app
# Function to convert a file to base64
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Function to set the background of the Streamlit app
def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{bin_str}");
        background-position: center;
        background-size: cover;
        font-family: "Times New Roman", serif;
    }}
    h1, h2, h3, p {{
        font-family: "Times New Roman", serif;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Set the background image for the app
set_background('2.jpeg')

# MDE Encryption and Decryption
def encrypt_mde(data):
    return base64.b64encode(data.encode()).decode()

def decrypt_mde(data):
    return base64.b64decode(data).decode()

# RSA Encryption and Decryption
def generate_rsa_keys():
    key = RSA.generate(2048)
    private_key = key.export_key()
    public_key = key.publickey().export_key()
    return private_key, public_key

def encrypt_rsa(public_key, data):
    rsa_key = RSA.import_key(public_key)
    cipher = PKCS1_OAEP.new(rsa_key)
    encrypted = cipher.encrypt(data.encode())
    return base64.b64encode(encrypted).decode()

def decrypt_rsa(private_key, encrypted_data):
    rsa_key = RSA.import_key(private_key)
    cipher = PKCS1_OAEP.new(rsa_key)
    encrypted_data = base64.b64decode(encrypted_data)
    decrypted = cipher.decrypt(encrypted_data).decode()
    return decrypted

# Streamlit App Title
st.title("Cryptographic System: Encryption & Decryption with Performance Analysis")

# Encryption & Decryption Section
st.subheader("Select Encryption Method")

encryption_method = st.selectbox("Select Encryption Method", ["MDE", "RSA"])
message = st.text_area("Enter the message to encrypt:")

if message:
    # Measure encryption and decryption time
    if encryption_method == "MDE":
        st.subheader("MDE Encryption")
        
        # Measure encryption time
        start_time = time.time()
        encrypted_message_mde = encrypt_mde(message)
        encryption_time_mde = time.time() - start_time
        st.write(f"Encrypted Message (MDE): {encrypted_message_mde}")
        st.write(f"Encryption Time (MDE): {encryption_time_mde:.4f} seconds")

        # Measure decryption time
        start_time = time.time()
        decrypted_message_mde = decrypt_mde(encrypted_message_mde)
        decryption_time_mde = time.time() - start_time
        st.write(f"Decrypted Message (MDE): {decrypted_message_mde}")
        st.write(f"Decryption Time (MDE): {decryption_time_mde:.4f} seconds")

    elif encryption_method == "RSA":
        st.subheader("RSA Encryption")
        private_key, public_key = generate_rsa_keys()

        # Measure encryption time
        start_time = time.time()
        encrypted_message_rsa = encrypt_rsa(public_key, message)
        encryption_time_rsa = time.time() - start_time
        st.write(f"Encrypted Message (RSA): {encrypted_message_rsa}")
        st.write(f"Encryption Time (RSA): {encryption_time_rsa:.4f} seconds")

        # Measure decryption time
        start_time = time.time()
        decrypted_message_rsa = decrypt_rsa(private_key, encrypted_message_rsa)
        decryption_time_rsa = time.time() - start_time
        st.write(f"Decrypted Message (RSA): {decrypted_message_rsa}")
        st.write(f"Decryption Time (RSA): {decryption_time_rsa:.4f} seconds")

import streamlit as st
import os
import pandas as pd
import base64
import subprocess
import time
import random
import string
from io import BytesIO
# from Crypto.PublicKey import RSA
from Crypto.PublicKey import RSA

from Crypto.Cipher import PKCS1_OAEP
import hashlib
import qrcode

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
set_background('7.jpg')


import streamlit as st
import random
import string
import hashlib
import qrcode
from io import BytesIO
import base64
import time
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP

# Function to generate a random OTP
def generate_otp():
    return ''.join(random.choices(string.digits, k=6))

# Password Hashing
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Function to generate QR code for UID
def generate_qr_code(uid):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(uid)
    qr.make(fit=True)
    img = qr.make_image(fill='black', back_color='white')

    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()

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
st.title("Cryptographic System: Login, OTP, Encryption & Decryption")

# User Authentication
st.subheader("User Authentication")

user_id = st.text_input("Enter User ID:")
password_input = st.text_input("Enter Password:", type="password")

# Hardcoded user credentials (password is hashed)
stored_user_id = "user1"
stored_password_hash = hash_password("password123")

if user_id == stored_user_id and hash_password(password_input) == stored_password_hash:
    st.success("Password is correct.")
    uid = "UID12345"  # Example Unique ID
    
    # Generate and display QR Code for UID
    qr_code_image = generate_qr_code(uid)
    st.image(qr_code_image, caption="Your QR Code", use_column_width=True)

    # OTP Generation (One-time)
    if 'otp' not in st.session_state:
        st.session_state.otp = generate_otp()
        st.session_state.otp_verified = False
        st.session_state.otp_attempted = False  # Track if OTP has been attempted

    st.write(f"Your OTP is: {st.session_state.otp}")
    
    otp_input = st.text_input("Enter OTP:")

    # OTP Verification
    if st.session_state.otp_verified:
        st.text_input("Enter OTP:", value=otp_input, disabled=True)
        
        # Display the message to proceed with encryption after OTP is verified
        st.write("Proceed with Encryption/Decryption below.")
        
        # Encryption/Decryption Options after OTP verification
        encryption_method = st.selectbox("Select Encryption Method", ["MDE", "RSA"])

        message = st.text_area("Enter the message to encrypt:")

        if encryption_method == "MDE":
            # MDE Encryption & Decryption
            st.subheader("MDE Encryption")
            with st.spinner("Encrypting message..."):
                start = time.time()
                encrypted_message_mde = encrypt_mde(message)
                encryption_time_mde = time.time() - start
                st.write(f"Encrypted Message (MDE): {encrypted_message_mde}")
                st.write(f"Encryption Time (MDE): {encryption_time_mde:.4f} seconds")

                start = time.time()
                decrypted_message_mde = decrypt_mde(encrypted_message_mde)
                decryption_time_mde = time.time() - start
                st.write(f"Decrypted Message (MDE): {decrypted_message_mde}")
                st.write(f"Decryption Time (MDE): {decryption_time_mde:.4f} seconds")

        elif encryption_method == "RSA":
            # RSA Encryption & Decryption
            st.subheader("RSA Encryption")
            private_key, public_key = generate_rsa_keys()

            with st.spinner("Encrypting message..."):
                start = time.time()
                encrypted_message_rsa = encrypt_rsa(public_key, message)
                encryption_time_rsa = time.time() - start
                st.write(f"Encrypted Message (RSA): {encrypted_message_rsa}")
                st.write(f"Encryption Time (RSA): {encryption_time_rsa:.4f} seconds")

                start = time.time()
                decrypted_message_rsa = decrypt_rsa(private_key, encrypted_message_rsa)
                decryption_time_rsa = time.time() - start
                st.write(f"Decrypted Message (RSA): {decrypted_message_rsa}")
                st.write(f"Decryption Time (RSA): {decryption_time_rsa:.4f} seconds")

    # If OTP is incorrect
    elif otp_input != "" and otp_input != st.session_state.otp and not st.session_state.otp_verified:
        if not st.session_state.otp_attempted:
            st.error("Incorrect OTP. Please try again.")
            st.session_state.otp_attempted = True  # Prevent further OTP attempts

    # If OTP is correct
    elif otp_input == st.session_state.otp and not st.session_state.otp_verified:
        st.session_state.otp_verified = True
        st.success("OTP Verified Successfully!")
        st.write("Proceed with Encryption/Decryption below.") 
        subprocess.run(["streamlit", "run", "app1.py"])# Make sure this is visible

else:
    st.error("Invalid User ID or Password.")


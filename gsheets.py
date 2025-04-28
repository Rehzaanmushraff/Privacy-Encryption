# gsheets.py
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import datetime

def connect_to_gsheet():
    # Update the scopes to modern versions
    scope = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive.file"
    ]
    
    creds = ServiceAccountCredentials.from_json_keyfile_name(
        "credentials.json", scope
    )
    client = gspread.authorize(creds)
    
    # Ensure exact match of spreadsheet name
    return client.open("EncryptionLogs").sheet1

def save_to_gsheet(original, encrypted, method):
    try:
        sheet = connect_to_gsheet()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sheet.append_row([timestamp, original, encrypted, method])
        st.success("✅ Successfully saved to Google Sheets!")
        return True
    except Exception as e:
        st.error(f"🔴 Google Sheets Error: {str(e)}")
        return False

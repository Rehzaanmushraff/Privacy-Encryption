# gsheets.py
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import datetime

def connect_to_gsheet():
    scope = ["https://spreadsheets.google.com/feeds",
             "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(
        "credentials.json", scope)
    client = gspread.authorize(creds)
    return client.open("EncryptionLogs").sheet1

def save_to_gsheet(original, encrypted, method):
    try:
        sheet = connect_to_gsheet()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sheet.append_row([timestamp, original, encrypted, method])
        return True
    except Exception as e:
        print(f"Google Sheets Error: {e}")
        return False

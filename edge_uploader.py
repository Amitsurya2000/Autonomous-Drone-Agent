import os
import time
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

# Authenticate Google Drive
gauth = GoogleAuth()
gauth.LocalWebserverAuth() # Opens browser to log in
drive = GoogleDrive(gauth)

# Your google drive folder id
DRIVE_FOLDER_ID = '1mzVbuOuXP9cGJksKTIFb8fnQZgf9amE5'
CROPS_DIR = 'data/crops'
uploaded_files = set()

print("[OK] Watching data/crops for new drone images...")

while True:
    # Check for new image
    for filename in os.listdir(CROPS_DIR):
        if filename.endswith('.jpg') and filename not in uploaded_files:
            file_path = os.path.join(CROPS_DIR, filename)
            print(f"[OK] New image found: {filename}, uploading to Google Drive...")
            
            # Upload to Google Drive
            # FIX: Change 'title': 'Drone_Alerts' to 'title': filename
            gfile = drive.CreateFile({'parents': [{'id': DRIVE_FOLDER_ID}], 'title': filename})
            gfile.SetContentFile(file_path)
            gfile.Upload()
            
            print(f"[OK] {filename} uploaded successfully!")
            uploaded_files.add(filename)
            
    time.sleep(5) # Check for new images every 5 seconds
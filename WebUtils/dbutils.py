import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from firebase_admin import storage
from firebase_admin import db
from firebase import firebase
import os
import json

DB_URL = "https://moska-377016-default-rtdb.europe-west1.firebasedatabase.app/"
STORAGE_URL = "moska-377016.appspot.com"
DB_REF : db.Reference = None
CRED : credentials.Certificate = None
APP : firebase_admin.App = None
FIRESTORE_CLIENT = None
STORAGE_BUCKET = None

def init_db():
    global CRED, APP, FIRESTORE_CLIENT, STORAGE_BUCKET, DB_REF
    # If everything is already initialized, then return
    if all([CRED, APP, FIRESTORE_CLIENT, STORAGE_BUCKET, DB_REF]):
        return
    CRED = credentials.Certificate("moska-admin.json")
    APP = firebase_admin.initialize_app(CRED, {
        'databaseURL': DB_URL,
        'storageBucket': STORAGE_URL
    })
    FIRESTORE_CLIENT = firestore.client()
    STORAGE_BUCKET = storage.bucket()
    DB_REF = db.reference()
    # If the database is empty, then populate it with the default data
    if DB_REF.get() is None:
        print("Database is empty, populating it with default data")
        try:
            # Email, username, password, experience level
            with open("default_user.json","r") as f:
                default_data = json.load(f)
            DB_REF.set(default_data)
            print("Database populated")
        except FileNotFoundError:
            print("Could not find default_user.json, database not populated")
    else:
        print("Database is not empty, not populating it")

def local_file_to_storage(local_file_path, storage_file_path):
    """ Upload a local file to the storage bucket """
    print(f"Uploading file '{local_file_path}' to storage '{storage_file_path}'")
    blob = STORAGE_BUCKET.blob(storage_file_path)
    blob.upload_from_filename(local_file_path)
    return

def list_blobs_in_folder(folder):
    """ List all files in the storage bucket, in the specified folder """
    print(f"Listing files in folder '{folder}'")
    blobs = list(STORAGE_BUCKET.list_blobs())
    blobs = [blob for blob in blobs if blob.name.startswith(folder)]
    print(f"Found blobs with names: {[blob.name for blob in blobs]}")
    return blobs


def get_gameid_from_folder(username, filename = ""):
    """ Find the next available game id in the folder.
    If the folder is empty, then return 0 """
    if not filename:
        filename = "Game-{x}-" +f"{username}" + ".log"
    if "{x}" not in filename:
        raise ValueError("Filename must contain '{x}'")
    pl_folder = f"{username}-Games/"
    #files = list(STORAGE_BUCKET.list_blobs(prefix=pl_folder))
    blobs = list_blobs_in_folder(pl_folder)
    # Find the next available game id
    game_id = 0
    while True:
        fname_to_check = pl_folder + filename.format(x=game_id)
        print(f"Checking if file '{fname_to_check}' exists")
        if not any([blob.name == fname_to_check for blob in blobs]):
            break
        game_id += 1
    print(f"Next available game id is {game_id}")
    return game_id
    



def add_user(email, username, password, exp_level):
    """ Add a user to the database, if they don't already exist. Returns True if the user was added, False otherwise """
    if get_user(username) is not None:
        print(f"User '{username}' already exists")
        return False
    print(f"Adding user '{username}'")
    DB_REF.push({
        "email": email,
        "username": username,
        "password": password,
        "exp_level": exp_level
    })
    return True

def get_user(username):
    """ Check if a user exists in the database. Returns the user if they exist, None otherwise """
    print(f"Retriving user {username}")
    users = DB_REF.get()
    # Traverse the nested dictionary to find the user
    for val, user in users.items():
        if not isinstance(user,dict):
            continue
        if user["username"] == username:
            print(f"Found user {user}")
            return user
    print(f"User not found")
    return None


def login_user(username, password):
    """ Login a user, if they exist. Returns the user if they exist, False otherwise """
    user = get_user(username)
    print(f"Logging in: {user}")
    if user is None:
        return False
    if user["password"] == password:
        print(f"Logged in user {user}")
        return user
    print(f"Password incorrect")
    return False



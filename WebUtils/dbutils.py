import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from firebase_admin import storage
from firebase_admin import db
from firebase import firebase
import os
import json
import datetime

DB_URL = "https://moska-377016-default-rtdb.europe-west1.firebasedatabase.app/"#os.environ["DB_URL"]
STORAGE_URL = "moska-377016.appspot.com"#os.environ["STORAGE_URL"]
DB_REF : db.Reference = None
CRED : credentials.Certificate = None
APP : firebase_admin.App = None
FIRESTORE_CLIENT = None
STORAGE_BUCKET = None
LOGGER = None

def initialize(logger):
    global LOGGER, DB_REF, CRED, APP, FIRESTORE_CLIENT, STORAGE_BUCKET
    LOGGER = logger
    if not all([DB_REF, STORAGE_BUCKET, FIRESTORE_CLIENT, APP, CRED]):
        init_db()
    logs_to_db()
    return

def logs_to_db():
    """ Every six hours, upload the logs to cloud.
    Logs are in the Logs folder.
    Dont upload the latest log file, as it is being written to.
    """
    LOGGER.info("Uploading logs to cloud")
    try:
        #Naming convention: "./Logs/" + "app"+datetime.datetime.now().strftime("%Y-%m-%d")+"-"+str(datetime.datetime.now().hour//6)+".log"
        # Get the latest log file
        latest_log_file = "app"+datetime.datetime.now().strftime("%Y-%m-%d")+"-"+str(datetime.datetime.now().hour//6)+".log"
        # Get all the log files
        log_files = os.listdir("./Logs")
        # Remove the latest log file
        log_files.remove(latest_log_file)
        if not log_files:
            LOGGER.info("No log files to upload")
            return
        # Upload all the log files
        for log_file in log_files:
            local_file_to_storage("./Logs/"+log_file, "Logs/"+log_file)
            os.remove("./Logs/"+log_file)
            LOGGER.info("Uploaded log file: "+log_file)
    except Exception as e:
        LOGGER.warning("Could not upload logs to cloud: "+str(e))
    return


def init_db():
    global CRED, APP, FIRESTORE_CLIENT, STORAGE_BUCKET, DB_REF
    # If everything is already initialized, then return
    if all([CRED, APP, FIRESTORE_CLIENT, STORAGE_BUCKET, DB_REF]):
        LOGGER.info("Database already initialized")
        return
    if not os.path.exists("moska-admin.json"):
        try:
            json_config_from_env_vars()
            LOGGER.info("Configured database from environment variables")
        except KeyError:
            print("Could not find moska-admin.json and could not configure database from environment variables")
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
        LOGGER.info("Database is empty, populating it with default data")
        #print("Database is empty, populating it with default data")
        try:
            # Email, username, password, experience level
            with open("default_user.json","r") as f:
                default_data = json.load(f)
            DB_REF.set(default_data)
            print("Database populated")
        except FileNotFoundError:
            print("Could not find default_user.json, database not populated")
    else:
        pass
    LOGGER.info("Database initialized")

def json_config_from_env_vars():
    """ Configure the database from environment variables"""
    config = {}
    config["type"] = os.environ["FIREBASE_TYPE"]
    config["project_id"] = os.environ["FIREBASE_PROJECT_ID"]
    config["private_key_id"] = os.environ["FIREBASE_PRIVATE_KEY_ID"]
    config["private_key"] = os.environ["FIREBASE_PRIVATE_KEY"]
    config["client_email"] = os.environ["FIREBASE_CLIENT_EMAIL"]
    config["client_id"] = os.environ["FIREBASE_CLIENT_ID"]
    config["auth_uri"] = os.environ["FIREBASE_AUTH_URI"]
    config["token_uri"] = os.environ["FIREBASE_TOKEN_URI"]
    config["auth_provider_x509_cert_url"] = os.environ["FIREBASE_AUTH_PROVIDER_X509_CERT_URL"]
    config["client_x509_cert_url"] = os.environ["FIREBASE_CLIENT_X509_CERT_URL"]
    # Save this config to a file, so that we can use it to initialize the app
    with open("moska-admin.json", "w") as f:
        json.dump(config, f)
    return


def local_file_to_storage(local_file_path, storage_file_path):
    """ Upload a local file to the storage bucket """
    #print(f"Uploading file '{local_file_path}' to storage '{storage_file_path}'")
    LOGGER.info(f"Uploading file '{local_file_path}' to storage '{storage_file_path}'")
    blob = STORAGE_BUCKET.blob(storage_file_path)
    blob.upload_from_filename(local_file_path)
    return

def list_blobs_in_folder(folder):
    """ List all files in the storage bucket, in the specified folder """
    blobs = list(STORAGE_BUCKET.list_blobs())
    blobs = [blob for blob in blobs if blob.name.startswith(folder)]
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
        if not any([blob.name == fname_to_check for blob in blobs]):
            break
        game_id += 1
    #print(f"Next available game id is {game_id}")
    LOGGER.info(f"Next available game id is {game_id}")
    return game_id

def add_user(email, username, password, exp_level):
    """ Add a user to the database, if they don't already exist. Returns True if the user was added, False otherwise """
    if get_user(username) is not None:
        #print(f"User '{username}' already exists")
        LOGGER.info(f"User '{username}' already exists")
        return False
    LOGGER.info(f"Adding user '{username}'")
    #print(f"Adding user '{username}'")
    DB_REF.push({
        "email": email,
        "username": username,
        "password": password,
        "exp_level": exp_level
    })
    return True

def get_user(username):
    """ Check if a user exists in the database. Returns the user if they exist, None otherwise """
    #print(f"Retriving user {username}")
    users = DB_REF.get()
    # Traverse the nested dictionary to find the user
    for val, user in users.items():
        if not isinstance(user,dict):
            continue
        if user["username"] == username:
            #print(f"Found user {user}")
            LOGGER.info(f"Found user {user}")
            return user
    LOGGER.info(f"User {username} not found")
    return None


def login_user(username, password):
    """ Login a user, if they exist. Returns the user if they exist, False otherwise """
    user = get_user(username)
    #print(f"Logging in: {user}")
    LOGGER.info(f"Logging in: {user}")
    if user is None:
        return False
    if user["password"] == password:
        #print(f"Logged in user {user}")
        LOGGER.info(f"Logged in user {user}")
        return user
    LOGGER.info(f"Password incorrect")
    #print(f"Password incorrect")
    return False


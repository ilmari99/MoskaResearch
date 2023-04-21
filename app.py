import random
import sys
import os
os.environ["PYTHONIOENCODING"] = "utf-8"
from flask import Flask, request, session, render_template
from flask_socketio import SocketIO, emit
import subprocess
import json
from flask_socketio import Namespace
from WebUtils import dbutils

app = Flask(__name__)
# Key is in a file APP-KEY
app.secret_key = os.environ["APP_KEY"]
socketio = SocketIO(app)
CARD_CONVERSION = json.load(open("./templates/card_conversions.json","r",encoding="utf-8"))
CARD_SUITS_TO_SYMBOLS = {"S":'♠', "D":'♦',"H": '♥',"C": '♣',"X":"X"}
CARD_SYMBOLS_TO_SUITS = {v:k for k,v in CARD_SUITS_TO_SYMBOLS.items()}


class GameNamespace(Namespace):
    def __init__(self, namespace):
        super().__init__(namespace=namespace)
        self.game_process = None
    
    def on_start_game(self):
        print(f"Starting game...")
        executable = sys.executable
        print(f"Executable: {executable}")
        game_id = dbutils.get_gameid_from_folder(session['username'])
        self.game_process = subprocess.Popen([str(executable), 'Play/play_in_browser.py', "--name", f"{session['username']}", "--gameid",str(game_id)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        universal_newlines=False)
        print(f"Game process created")
        for line in iter(self.game_process.stdout.readline, b''):
            line = line.decode("utf-8")
            # If users name is in line, then bold it
            if session['username'] in line:
                line = line.replace(session['username'], f"<b>{session['username']}</b>")
            # Possible cards are separated by a space or a comma
            poss_cards = line.split(" ")
            poss_cards = [pcard.split(",") for pcard in poss_cards]
            poss_cards = [card for sublist in poss_cards for card in sublist]
            poss_cards = [card.strip() for card in poss_cards]
            poss_cards = [c.replace("[","").replace("]","").replace("{","").replace("}","").replace(":","") for c in poss_cards]
            print(f"Possible cards: {poss_cards}")
            for card in filter(lambda x : True if x else False, poss_cards):
                if card in CARD_CONVERSION:
                    if card == "-1X":
                        line = line.replace(card, f"<img src=\"{CARD_CONVERSION[card]}\" alt=\"X\" height=\"50\" width=\"30\">")
                    else:
                        # Replace the card with the unicode symbol and html image tag
                        line = line.replace(card, f"<img src=\"{CARD_CONVERSION[card]}\" alt=\"X\" height=\"65\" width=\"40\">",1)
            line = line.replace("[","").replace("]","")
            print(f"New line: {line}")
            self.emit('output', line)
        self.game_process.wait()
        pl_folder = f"{session['username']}-Games"
        print(f"Writing to storage...")
        for file in os.listdir(pl_folder):
            if f"Game-{game_id}" in file:
                dbutils.local_file_to_storage(f"{pl_folder}/{file}", f"{pl_folder}/{file}")
                os.remove(f"{pl_folder}/{file}")
            elif file == "Vectors":
                p = f"{pl_folder}/{file}"
                vectors = os.listdir(p)
                if vectors:
                    file = f"{pl_folder}/{file}/{os.listdir(p)[0]}"
                    dbutils.local_file_to_storage(file, file)
                    os.remove(file)
        self.emit('output', 'Thank you for playing! For a new game refresh the page.')

    def on_input(self, input):
        self.game_process.stdin.write((input + '\n').encode())
        self.game_process.stdin.flush()

    def on_disconnect(self):
        print(f"Disconnecting...")
        if self.game_process is not None and self.game_process.poll() is None:
            self.game_process.stdin.write(('exit\n').encode())
            self.game_process.stdin.flush()
            print(f"Killed game process",flush=True)
        else:
            print(f"No game process to kill",flush=True)


@app.route('/')
def home():
    dbutils.init_db()
    if 'username' in session:
        return render_template('home.html')
    else:
        return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        exp_level = request.form['experience']
        email = request.form['email']
        suc = dbutils.add_user(email=email, username=username, password=password,exp_level=exp_level)
        if suc:
            return render_template('login.html')
        return 'Registration failed. Username might be taken.'
    else:
        return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = dbutils.login_user(username, password)
        if user:
            session['username'] = username
            return render_template('home.html')
        else:
            return render_template('login.html')
    else:
        return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return render_template('index.html')

@app.route('/play_game')
def play_game():
    #print(f"Starting game maybe")
    namespace = f"/game-{session['username']}-{str(random.randint(0,1000000))}"
    socketio.on_namespace(GameNamespace(namespace))
    return render_template('play_game.html', namespace=namespace)


if __name__ == '__main__':
    app.run(debug=False)
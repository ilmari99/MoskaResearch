import sys
import os
os.environ["PYTHONIOENCODING"] = "utf-8"
from flask import Flask, request, session, render_template
from flask_socketio import SocketIO, emit
import subprocess
import sqlite3
import json
app = Flask(__name__)
# Key is in a file APP-KEY
app.secret_key = open("APP-KEY","r").read()
socketio = SocketIO(app)
CARD_CONVERSION = json.load(open("./templates/card_conversions.json","r",encoding="utf-8"))
CARD_SUITS_TO_SYMBOLS = {"S":'♠', "D":'♦',"H": '♥',"C": '♣',"X":"X"}
CARD_SYMBOLS_TO_SUITS = {v:k for k,v in CARD_SUITS_TO_SYMBOLS.items()}


def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT, password TEXT)''')
    conn.commit()
    conn.close()

@app.route('/')
def home():
    if 'username' in session:
        return render_template('home.html')
    else:
        return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect('users.db')
        c = conn.cursor()

        c.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = c.fetchone()

        if user:
            return 'Username already exists'

        c.execute('INSERT INTO users VALUES (?, ?)', (username, password))
        conn.commit()
        conn.close()

        return 'Registration successful'
    else:
        return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect('users.db')
        c = conn.cursor()

        c.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password))
        user = c.fetchone()

        conn.close()

        if user:
            session['username'] = user[0]
            return render_template('home.html')
        else:
            return render_template('login.html')
    else:
        return render_template('login.html')

@app.route('/logout')
def logout():
    if "game_process" in globals() and game_process is not None and game_process.poll() is None:
        game_process.stdin.write(('exit\n').encode())
        game_process.stdin.flush()
        print(f"Killed game process",flush=True)
    session.pop('username', None)
    return render_template('index.html')

@app.route('/play_game')
def play_game():
    print(f"Starting game maybe")
    return render_template('play_game.html')

@socketio.on('start_game')
def handle_start_game():
    print(f"Starting game...")
    global game_process
    executable = sys.executable
    print(f"Executable: {executable}")
    game_process = subprocess.Popen([str(executable), 'Play/play_in_browser.py', "--name", f"{session['username']}"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    universal_newlines=False)
    print(f"Game process created")
    session['game_process'] = game_process
    for line in iter(game_process.stdout.readline, b''):
        line = line.decode("utf-8")
        # If users name is in line, then bold it
        if session['username'] in line:
            line = line.replace(session['username'], f"<b>{session['username']}</b>")
        # Possible cards are separated by a space or a comma
        poss_cards = line.split(" ")
        poss_cards = [pcard.split(",") for pcard in poss_cards]
        poss_cards = [card for sublist in poss_cards for card in sublist]
        poss_cards = [card.strip() for card in poss_cards]
        poss_cards = [c.replace("[","").replace("]","").replace("{","").replace("}","") for c in poss_cards]
        print(f"Possible cards: {poss_cards}")
        for card in filter(lambda x : True if x else False, poss_cards):
            if card in CARD_CONVERSION:
                if card == "-1X":
                    line = line.replace(card, f"<img src=\"{CARD_CONVERSION[card]}\" alt=\"X\" height=\"50\" width=\"30\">")
                else:
                    # Replace the card with the unicode symbol and html image tag
                    line = line.replace(card, f"<img src=\"{CARD_CONVERSION[card]}\" alt=\"X\" height=\"65\" width=\"40\">",1)
        line = line.replace("[","").replace("]","").replace(",","")
        print(f"New line: {line}")
        emit('output', line)
    game_process.wait()
    emit('output', 'Game over')

@socketio.on('input')
def handle_input(input):
    game_process.stdin.write((input + '\n').encode())
    game_process.stdin.flush()



if __name__ == '__main__':
    init_db()
    app.run(debug=False)

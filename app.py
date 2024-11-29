from flask import Flask, request, render_template, jsonify
from module import MTGProject

app = Flask(__name__)

mtg = MTGProject()

@app.route("/")
def hello_world():
	return render_template('bruh.html')

@app.route('/bruh')
def bruh():
	return render_template('bruh.html')


@app.route('/getcard', methods = ['POST'])
def get_card():
	data = request.get_json()
	cardname = data['cardname']
	card = mtg.get_card(cardname)
	return jsonify(card)
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

@app.route("/")
def hello_world():
    return 'Index Page'

@app.route('/hello')
def hello():
    return render_template('hello.html')

@app.route('/bruh')
def bruh():
    return render_template('bruh.html')

@app.route('/calculate', methods = ['POST'])
def calc_square():
    number = float(request.form['number'])
    square = number ** 2
    return jsonify({"result": square})

@app.route('/getcard', methods = ['POST'])
def get_card():
    data = request.get_json() 
    cardname = data['cardname']
    print(f'card: {cardname}')
    return jsonify({"cardname": cardname})
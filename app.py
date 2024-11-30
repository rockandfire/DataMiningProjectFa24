from flask import Flask, request, render_template, jsonify
from module import MTGProject


#to run: python3 -m flask --app app run


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

@app.route('/getupcomingcard', methods = ['POST'])
def get_upcoming_card():
	data = request.get_json()
	cardname = data['cardname']
	card = mtg.get_upcoming_card_data(cardname)
	return jsonify(card)

@app.route('/submitcommander', methods = ['POST'])
def submit_commander():
	data = request.get_json()
	commander_name = data['commander_name']
	print(f'recieved commander: {commander_name}')

	# begins the model training
	mtg.get_related_cards(commander_name)
	mtg.compute_upcoming_recommendations(commander_name)
	related_cards = mtg.get_related_card_names()
	upcoming_cards = mtg.get_upcoming_card_names()
	returnObj = {'related_cards': related_cards,
				 'upcoming_cards': upcoming_cards}
	return jsonify(returnObj)

@app.route('/search_commander', methods=['POST'])
def search_commander():
    data = request.get_json()
    search_term = data['search']
    
    # Just search for creatures for now
    matching_creatures = mtg.df_cards[
        (mtg.df_cards['name'].str.contains(search_term, case=False, na=False)) &
        (mtg.df_cards['types'].str.contains('Creature', case=False, na=False))
    ]
    
    # Return all matching creature names
    matching_names = matching_creatures['name'].tolist()
    return jsonify({'commanders': matching_names[:10]})
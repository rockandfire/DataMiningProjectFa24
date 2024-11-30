from flask import Flask, request, jsonify
from flask_cors import CORS
from testing import MTGDeckBuilder
import numpy as np
import math
import json
import pandas as pd

app = Flask(__name__)
CORS(app)

deck_builder = MTGDeckBuilder(debug=False)

def convert_sets_to_lists(obj):
    """Convert any sets in the object to lists for JSON serialization"""
    if isinstance(obj, dict):
        return {key: convert_sets_to_lists(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_sets_to_lists(item) for item in obj]
    elif isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif pd.isna(obj):  # Handle pandas NA/NaN
        return None
    return obj

@app.route('/api/commanders', methods=['GET'])
def get_commanders():
    search_term = request.args.get('search', '').lower()
    commanders = deck_builder.get_valid_commanders()
    filtered_commanders = commanders[commanders.str.lower().str.contains(search_term)]
    return jsonify(filtered_commanders.tolist())

@app.route('/api/analyze-commander', methods=['POST'])
def analyze_commander():
    try:
        data = request.json
        commander = data.get('commander')
        if not commander:
            return jsonify({'error': 'No commander specified'}), 400
            
        results = deck_builder.analyze_deck(commander)
        serializable_results = convert_sets_to_lists(results)
        
        # Debug print
        print("API Response:", json.dumps(serializable_results, indent=2))
        
        return jsonify(serializable_results)
    except Exception as e:
        print(f"Error in analyze_commander: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/api/analyze-future-card', methods=['POST'])
def analyze_future_card():
    try:
        data = request.json
        
        # Validate inputs
        if not data.get('cardName'):
            return jsonify({'error': 'Card name is required'}), 400
            
        mana_value = data.get('manaValue')
        if mana_value:
            try:
                mana_value = float(mana_value)
            except ValueError:
                return jsonify({'error': 'Invalid mana value'}), 400
        else:
            mana_value = 0
            
        results = deck_builder.analyze_future_card(
            card_name=data.get('cardName', ''),
            card_text=data.get('cardText', ''),
            mana_value=mana_value,
            color_identity=data.get('colorIdentity', [])
        )
        
        serializable_results = convert_sets_to_lists(results)
        
        # Debug print
        print("API Response:", json.dumps(serializable_results, indent=2))
        
        return jsonify(serializable_results)
    except Exception as e:
        print(f"Error in analyze_future_card: {str(e)}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
import pandas as pd
import json
import numpy as np
import sys
from sklearn.model_selection import train_test_split
import random

#basically is converting this file to a class so that it's easier to deal with during gui runtime
class MTGProject:
	def __init__(self):
		#initializes the cards 
		

		cards = pd.read_csv('./cards.csv')
		cards = cards[cards.availability.str.contains('paper')].drop_duplicates(subset='name', keep='last')
		cards = cards[~cards.types.str.contains('Land')]
		cards = cards[~cards.text.isnull()]
		cards = cards[['name', 'colorIdentity', 'keywords', 'text', 'edhrecRank', 'manaValue', 'uuid', 'availability', 'isOnlineOnly', 'isTextless', 'manaCost', 'types']]
		cards['edhrecRank'] = cards['edhrecRank'].fillna(cards['edhrecRank'].max())

		self.df_cards = cards

	#gets card info from dataset
	#mainly used for gui
	def get_card(self, cardname):
		row = self.df_cards.loc[self.df_cards['name'] == 'Grey Knight Paragon'].iloc[0]
		card_text = row['text']
		card_text = card_text.replace('\\n', '- ')
		print(card_text)

		card = {
			'cardname': cardname,
			'types': row['types'],
			'manacost': row['manaCost'],
			#'cardtext': row['text'],
			'cardtext': card_text,
			'edhrec': row['edhrecRank']
			} 
		return card
import pandas as pd
import json
import numpy as np
import sys
from sklearn.model_selection import train_test_split
import random

def filter_identity(commander_name, cards):
	commander_identity = set(cards[cards.name == commander_name]['colorIdentity'].iloc[0].split(', '))
	# Get all indices of compliant cards
	sum = 0
	indices = []
	# print(cards.index)

	# print(cards.index[0])
	# print(cards.loc[2684]['colorIdentity'])
	for row in cards.index:
		if type(cards.loc[row]['colorIdentity']) is float:
			identity = set()
		else:
			identity = set(cards.loc[row]['colorIdentity'].split(', '))
		if identity <= commander_identity:
			indices.append(row)
			sum += 1

	valid_cards = cards[cards.index.isin(indices)]
	return valid_cards

def extract_text_perms(card):
	
	# If any ability in the catalyst triggers an ability in the reagent
	# append the pair of indices to the edge list
	rules = card['text'].replace('\\n', ' ')
	# For each rule, split each word and find all combinations of rules text in order.
	# 
	rule_phrases = []
	rule_components = rules.lower().strip().replace(',', '').replace('.', '').replace(':', '').split(' ')

	# For each word in parsed phrase, join next j words until end of phrase
	for i in range(len(rule_components)):
		for j in range(i + 1, len(rule_components) + 1):
			rule_phrases.append(" ".join(rule_components[i:j]))

	return rule_phrases

def get_similarity(card_a, card_b):
	# print(set(rule_phrases[card_a]))
	return len(set(rule_phrases[card_a]).intersection(set(rule_phrases[card_b]))) / len(rule_phrases[card_b])

# Perform K-Means clustering on the set of valid 
# cards using EDHRec Rank to determine distance
def k_means(k, cards):
	centroids = []

	edh_scores = sorted(cards['edhrecRank'].to_list())

	# Maybe update to force into range of quarters of true range?
	for i in range(k):
		centroids.append(np.random.randint(edh_scores[(i * len(edh_scores)) // k] , edh_scores[(((i + 1) * len(edh_scores)) // k) - 1]))

	# Compute minimum distance for each card and assign accordingly
	assignments = []
	for i in range(k):
		cluster = []
		assignments.append(cluster)

	for index in cards.index:
		min_distance = sys.maxsize
		assigned_centroid_index = centroids.index(max(centroids))

		for rank in centroids:
			if abs(cards.loc[index]['edhrecRank'] - rank) < min_distance:
				assigned_centroid_index = centroids.index(rank)
				min_distance = abs(cards.loc[index]['edhrecRank'] - rank)
		assignments[assigned_centroid_index].append(index)

	# Compute new centroids based on averages of cards in cluster
	clustered_cards = []
	for i in range(k):
		
		sum = 0
		for index in assignments[i]:
			sum += cards.loc[index]['edhrecRank']
		
		centroids[i] = sum / len(assignments[i])
		clustered_cards.append((centroids[i], assignments[i]))
	clustered_cards = sorted(clustered_cards, key=lambda x:x[0])
	
	df_clusters = []
	for cluster in clustered_cards:
		df_clusters.append(cards[cards.index.isin(cluster[1])])
	return df_clusters

# Take a list of dataframes containing cards clustered based 
# on EDHRec rank. Starting with the most competitive cards, 
# perform modified Apriori until the threshold of 63 cards
# is met. The remaining 37 cards will be basic lands
def apriori(commander_name, clusters, cards):    

	# Iterate over each cluster until length requirement is met
	# When out of unique cards in cluster, move to next
	# Select the single candidate with the highest average support 
	# among all cards in current deck, then continue
	cluster_pos = 0
	# commander_index = cards[cards.name == commander_name].index[0]
	# commander_similarity = 
	current_deck = [cards[cards.name == commander_name].index[0]]
	sim_card = [cards[cards.name == commander_name].index[0]]
	similarities = [1]
	while cluster_pos < len(clusters) and len(current_deck) < 63:
		# Compute similarity of a given card to each card in list
		# Grab given card
		for card in current_deck[:(len(current_deck) // 50) + 1]:
			# Get average similarity across list?
			max_similarity = -1
			max_index_a = sys.maxsize
			max_index_b = sys.maxsize
			for index in clusters[cluster_pos].index:

				# get similarity between current card and each card already in deck
				card_similarity = get_similarity(card, index)
				if card_similarity > max_similarity and index not in current_deck:
					# print(card_similarity)
					max_similarity = card_similarity 
					max_index_a = card
					max_index_b = index
			# Append support with max indices
			# Compute confidence by stringing together a pair 
			# with the card containing next highest support value, 
			# then compute standard support
			
		if len(current_deck) > 63:
			break

		sim_card.append(max_index_a)
		current_deck.append(max_index_b)
		similarities.append(max_similarity)
			# print(len(current_deck))
		if max_index_b == clusters[cluster_pos].index[-1]:
			cluster_pos += 1
	# print(max(candidate_cards, key=lambda item: item[0]), cards.loc[max(candidate_cards, key=lambda item: item[0])[1]])

	rules = []
	for i in range(len(current_deck)):
		rules.append((sim_card[i], current_deck[i], similarities[i], cards.loc[current_deck[i]]['edhrecRank']))

	rules_2 = sorted(rules, key=lambda x:x[3])

	# Using rules of size 2, generate rules of size 3 and calculate support (similarity) and confidence
	
	rules = []
	for card in current_deck:
		max_similarity = -1
		max_index_a = sys.maxsize
		max_index_b = sys.maxsize
		for rule in rules_2:	
			# Get max similarity of card to either card in rule
			card_similarity = get_similarity(rule[1], card)
			if card_similarity > max_similarity:
				# print(card_similarity)
				max_similarity = card_similarity 
				max_index_a = rule[0]
				max_index_b = rule[1]
				rule_2_similarity = rule[2]
		# Find denom for support by taking average similarity for included over entire deck
		sum = 0
		for included_card in current_deck:
			sum += get_similarity(included_card, card)
		denom = sum / len(current_deck)

		# Take sum of rule similarity and max / denom
		support_3 = (card_similarity + rule_2_similarity) / denom

		# Take sum of rule similarity and max / similarity to compute confidence for 3
		confidence = (card_similarity + rule_2_similarity) / rule_2_similarity
		rules.append((rule[0], rule[1], card, support_3, confidence, cards.loc[card]['edhrecRank']))

	rules_3 = sorted(rules, key=lambda x:x[5])

	return rules_2, rules_3

class SimpleLinearRegression:
	def __init__(self):
		#initializes variables
		self.slope_ = None
		self.intercept_ = None
		
	def fit(self, x, y):
		#calculates the mean of the input and labels
		Xmean = np.mean(x)
		ymean = np.mean(y)
		
		#calculate terms needed for slope and intercept of regression line
		numerator = np.sum((x - Xmean) * (y - ymean))
		denominator = np.sum((x - Xmean) ** 2)

		#calculate slope and intercept of regression line
		self.slope_ = numerator / denominator
		self.intercept_ = ymean - self.slope_ * Xmean
		
	def predict(self, x):
		return self.intercept_ + self.slope_ * x
	
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
cards = pd.read_csv('./cards.csv')
cards = cards[cards.availability.str.contains('paper')].drop_duplicates(subset='name', keep='last')
cards = cards[~cards.types.str.contains('Land')]
cards = cards[~cards.text.isnull()]
cards = cards[['name', 'colorIdentity', 'keywords', 'text', 'edhrecRank', 'manaValue', 'uuid', 'availability', 'isOnlineOnly', 'isTextless']]
cards['edhrecRank'] = cards['edhrecRank'].fillna(cards['edhrecRank'].max())

#drops ehrec ranks that don't exist
df = cards[cards.edhrecRank != -1]
#converts edhrec to a np array and then creates a random sim score for each row loosely based on edhrec
df_edhrec = np.array(df['edhrecRank'].tolist())
df_sim_score = [0.75 * x + random.gauss(0, 10) for x in df_edhrec]
#splits data into test and validation sets, 75% training, 25% validation
Xtrain, Xval, ytrain, yval = train_test_split(df_edhrec, df_sim_score, test_size=0.25, random_state=42)

#converts data to dataframes
Xtrain = pd.DataFrame(Xtrain)
Xval = pd.DataFrame(Xval)
ytrain = pd.DataFrame(ytrain)
yval = pd.DataFrame(yval)

#simple test of
model = SimpleLinearRegression()
model.fit(Xtrain, ytrain)
preds = model.predict(Xval)

#performs K fold cross validation on linear regression model
def k_fold_cross_validation(model, Xtrain, ytrain, k):

	#convert to numpy array for np functions
	Xtrain = np.array(Xtrain)
	ytrain = np.array(ytrain)
	
	#shuffle data
	indices = np.arange(Xtrain.shape[0])
	np.random.shuffle(indices)
	Xtrain, ytrain = Xtrain[indices], ytrain[indices]
	
	#split data into folds
	fold_size = len(Xtrain) // k
	accuracies = []
	
	for i in range(k):
		#split ddata into val and training sets
		start, end = i * fold_size, (i + 1) * fold_size
		X_val_fold = Xtrain[start:end]
		y_val_fold = ytrain[start:end]
		
		X_train_fold = np.concatenate([Xtrain[:start], Xtrain[end:]])
		y_train_fold = np.concatenate([ytrain[:start], ytrain[end:]])
		
		#fit model with selected data
		model.fit(X_train_fold, y_train_fold)
		
		#eval current model
		predictions = model.predict(X_val_fold)
		#calculate mse
		mse = np.mean((predictions - y_val_fold) ** 2)
		accuracies.append(mse)
	
	mean_accuracy = np.mean(accuracies)
	return mean_accuracy

#prints predicted values versus actual values
for x in range(5):
	print(f'predicted: {preds.iloc[x, 0]}, actual: {yval.iloc[x, 0]}')

print("Total card count for commander legality: {}".format(len(cards.index)))

with open('Keywords.json') as json_file:
	keywords = json.load(json_file)

#load dict of all keywords
keywords = pd.DataFrame({'keywords': keywords})

# Define commander name and filter for color identity
commander_name = "Yuriko, the Tiger's Shadow"
cards = filter_identity(commander_name, cards)

print("Total number of cards legal for deck: {}".format(len(cards.index)))

# Extract all rules phrases for apriori comparison
rule_phrases = {}
for i in cards.index:
	rule_phrases[i] = extract_text_perms(cards.loc[i])

# Perform K-Means clustering for k clusters on cards
clusters = k_means(12, cards)

# Print length of each cluster
for i in range(len(clusters)):
	print("Number of cards in cluster {}: {}".format(i, len(clusters[i])))

# Generate rules of length 2 for deck
rules_2, rules_3 = apriori(commander_name, clusters, cards)

# Print each rule
# for rule in rules_2:
# 	print("Most similar card: {}\nAdded card: {}\nSimilarity Score: {}\nEDHRec Rank: {}\n".
# 	format(cards.loc[rule[0]]['name'], cards.loc[rule[1]]['name'], rule[2], rule[3]))

# for rule in rules_3:
# 	print("Card 1: {}\nCard 2: {}\nCard 3: {}\nSimilarity Score: {}\nConfidence Score: {}\nEDHRec Rank: {}\n".
# 	format(cards.loc[rule[0]]['name'], cards.loc[rule[1]]['name'], cards.loc[rule[2]]['name'], rule[3], rule[4], rule[5]))

model = SimpleLinearRegression()
mean_mse = k_fold_cross_validation(model, Xtrain, ytrain, 5)
avg = np.mean(ytrain)
print(f"MSE from k-fold cross-validation: {mean_mse}")
print(f"average edhrec value of training data: {avg}")
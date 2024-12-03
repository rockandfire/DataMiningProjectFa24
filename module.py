import pandas as pd
import json
import numpy as np
import sys
from sklearn.model_selection import train_test_split
import random
from LinearRegression import SimpleLinearRegression 
import math

# Monolithic application class for easy access by frontend
class MTGProject:

    # Initialize cards based on availability and set code
    # Training and test data are stored in separate data frames
    def __init__(self, upcoming_code = 'DSK'):
		
        # Preprocess main data frame to remove irrelevant cards
        cards = pd.read_csv('./cards.csv', low_memory=False)
        cards = cards[cards.availability.str.contains('paper')].drop_duplicates(subset='name', keep='last')
        cards = cards[~cards.types.str.contains('Land')]
        cards = cards[~cards.text.isnull()]
        cards = cards[['name', 'colorIdentity', 'keywords', 'text', 'edhrecRank', 'manaValue', 'uuid', 'availability', 'isOnlineOnly', 'isTextless', 'manaCost', 'types', 'setCode']]
        cards['edhrecRank'] = cards['edhrecRank'].fillna(cards['edhrecRank'].max())

        self.df_cards = cards

        # Split data frame into upcoming and current cards
        self.df_current_cards = cards[~cards['setCode'].str.contains(upcoming_code)]
        self.df_upcoming_cards = cards[cards['setCode'].str.contains(upcoming_code)]

        # Initializes other variables for later functions
        self.df_related_commander_cards = None
        self.clusters = None
        self.rule_phrases = []
        self.df_related_cards = None
        self.upcoming_rec_cards = None
        self.upcoming_indices = None
        self.linear_model = None
        self.upcoming_cards_data = None

	# Returns card info from dataset
	# Mainly used for GUI
    def get_card(self, cardname):

        # Get entry matching card name
        row = self.df_cards.loc[self.df_cards['name'] == cardname].iloc[0]

        # Removes annoying \n in the text
        card_text = row['text']
        card_text = card_text.replace('\\n', '- ')

        # Store card info in a dict for easy access
        card = {
            'cardname': cardname,
            'types': row['types'],
            'manacost': row['manaCost'],
            'cardtext': card_text,
            'edhrec': row['edhrecRank'],

            # This info is used for within class and not in GUI
            'setcode': row['setCode']
            } 
        return card
    
    # Perform K-Means clustering on the set of valid 
    # cards using EDHRec Rank to determine distance
    def k_means(self, k, cards_in_k):

        # Declare empty centroid list and sort cards by EDHRec Rank
        centroids = []
        edh_scores = sorted(cards_in_k['edhrecRank'].to_list())

        # Force pseudorandom centroid generation to 
        for i in range(k):

            # Prevent index out of bounds by assigning max bound on last cluster
            max_range = edh_scores[(((i + 1) * len(edh_scores)) // k) - 1]
            if i == k - 1:
                max_range = max(edh_scores)
            centroids.append(np.random.randint(edh_scores[(i * len(edh_scores)) // k], max_range))

        # Compute minimum distance for each card and assign accordingly
        assignments = []
        for i in range(k):
            cluster = []
            assignments.append(cluster)

        # Assign each card by storing its index in the data frame to a centroid
        for index in cards_in_k.index:

            # Reset minimum distance each loop
            min_distance = sys.maxsize
            assigned_centroid_index = centroids.index(max(centroids))

            # Compare card rank to each centroid to find distance
            for rank in centroids:

                # Recompute min distance to find correct centroid
                if abs(cards_in_k.loc[index]['edhrecRank'] - rank) < min_distance:
                    assigned_centroid_index = centroids.index(rank)
                    min_distance = abs(cards_in_k.loc[index]['edhrecRank'] - rank)
            
            # Assign card index to cluster based on result
            assignments[assigned_centroid_index].append(index)

        # Compute new centroids based on averages of cards in cluster
        clustered_cards = []
        for i in range(k):
            
            # Compute average EDHRec Rank of cluster
            rank_sum = 0
            for index in assignments[i]:
                rank_sum += cards_in_k.loc[index]['edhrecRank']
            
            centroids[i] = rank_sum / len(assignments[i])

            # Store as tuple of centroid and assigned cards
            clustered_cards.append((centroids[i], assignments[i]))

        # Sort clusters by centroid to move highly ranked cards to top of list
        clustered_cards = sorted(clustered_cards, key=lambda x:x[0])
        
        # Create list of data frames containing each card
        df_clusters = []
        for cluster in clustered_cards:
            df_clusters.append(cards_in_k[cards_in_k.index.isin(cluster[1])])

        # Internal performance analysis using SSE
        sse_clusters = []
        for cluster in clustered_cards:

            # Compute sum of squared error for each cluster
            sse = 0
            for index in cluster[1]:
                sse += math.pow(int(cards_in_k.loc[index]['edhrecRank']) - cluster[0], 2)
            sse_clusters.append(sse)

        # SSE is found to be a poor metric for performance due to 
        # the highly scattered nature of the data paired with 
        # the intention behind applying K Means to this stage of 
        # the solution
        print('SSE for each cluster: {}'.format(sse_clusters))

        # Print SSE per cluster (known large)
        for i in range(len(sse_clusters)):
            sse_clusters[i] = sse_clusters[i] - (sum(sse_clusters) / len(sse_clusters))
        print('SSE Comparison against cluster average: {}'.format(sse_clusters))
        
        # Rather than use SSE, compute distance of lowest ranked 
        # card to centroid to measure performance. Aligns more 
        # closely with the intended functionality, allowing 
        # for simpler verification
        sse_clusters = []
        for i in range(len(clustered_cards) - 1):
            dist_current = int(cards_in_k.loc[clustered_cards[i][1][-1]]['edhrecRank']) - clustered_cards[i][0]
            dist_lower = int(cards_in_k.loc[clustered_cards[i][1][-1]]['edhrecRank']) - clustered_cards[i + 1][0]
            sse_clusters.append((dist_current, dist_lower))

        # Print all distances to confirm item belongs in centroid 
        print('Distance of lowest ranked card from current centroid to next:')
        
        for i in range(len(sse_clusters)):
            print('Cluster {}: {} vs. {}'.format(i, abs(sse_clusters[i][0]), abs(sse_clusters[i][1])))

        # Return data frames corresponding to clusters
        return df_clusters
    
	# Filters data frame for cards related to color identity of given Commander
    def filter_identity(self, commander_name):

        # Perform string manipulation to determine color identity
        cards = self.df_cards
        commander_identity = set(cards[cards.name == commander_name]['colorIdentity'].iloc[0].split(', '))
        
        # Handle NaN (colorless) cards accordingly and perform
        # set comparison to filter cards
        sum = 0
        indices = []
        for row in cards.index:

            # Convert NaN to empty set
            if type(cards.loc[row]['colorIdentity']) is float:
                identity = set()

            # Otherwise, string manip to get color identity
            else:
                identity = set(cards.loc[row]['colorIdentity'].split(', '))

            # Add index to list if card identity is a subset of Commander colors
            if identity <= commander_identity:
                indices.append(row)
                sum += 1

        # Filter valid cards based on indices in color identity
        return cards[cards.index.isin(indices)]
	
	# For every term in card rules text, append 
    # all phrases to a list for similarity (support)
    # comparison
    def extract_text_perms(self, card):

        # Split abilities based on newline character
        rules = card['text'].replace('\\n', ' ')

        # For each rule, split each word and 
        # find all in-order combinations of rules text
        rule_phrases = []
        rule_components = rules.lower().strip().replace(',', '').replace('.', '').replace(':', '').split(' ')

        # For each word in parsed phrase, 
        # join next j words until end of phrase
        for i in range(len(rule_components)):
            for j in range(i + 1, len(rule_components) + 1):
                rule_phrases.append(" ".join(rule_components[i:j]))

        # Return list of all phrases for given card
        return rule_phrases
    
    # Given a pair of cards, access their 
    # dict entry for extracted phrases
    # to obtain their similarity (support)
    def get_similarity(self, card_a, card_b):

        # Return simple set intersection of rule phrases
        return len(set(self.rule_phrases[card_a]).intersection(set(self.rule_phrases[card_b]))) / len(self.rule_phrases[card_b])
    

    # Take a list of dataframes containing cards clustered based 
    # on EDHRec rank. Starting with the most competitive cards, 
    # perform modified Apriori until the threshold of ~63 cards
    # is met. The remaining ~37 cards will be basic lands
    def apriori(self, commander_name, clusters, cards):    

        # Iterate over each cluster until length requirement is met
        # When out of unique cards in cluster, move to next
        # Select the single candidate with the highest average support 
        # among all cards in current deck, then continue
        cluster_pos = 0
        current_deck = [cards[cards.name == commander_name].index[0]]
        sim_card = [cards[cards.name == commander_name].index[0]]
        similarities = [1]
        while cluster_pos < len(clusters) and len(current_deck) < 63:

            # Compute similarity of a given card to each card in top list
            for card in current_deck[:(len(current_deck) // 50) + 1]:
                # Get average similarity across list?
                max_similarity = -1
                max_index_a = sys.maxsize
                max_index_b = sys.maxsize
                for index in clusters[cluster_pos].index:

                    # get similarity between current card and each card already in deck
                    card_similarity = self.get_similarity(card, index)
                    if card_similarity > max_similarity and index not in current_deck:
                        max_similarity = card_similarity 
                        max_index_a = card
                        max_index_b = index
            
            # Break out when deck length limit met
            if len(current_deck) >= 63:
                break
            
            # Update all lists
            sim_card.append(max_index_a)
            current_deck.append(max_index_b)
            similarities.append(max_similarity)

            # Check if maximally similar card is the last card in
            # cluster. If so, move to next cluster
            if max_index_b == clusters[cluster_pos].index[-1]:
                cluster_pos += 1

        # Store generated rules in a tuple containing cards, support, and EDHRec of newly included card
        rules = []
        for i in range(len(current_deck)):
            rules.append((sim_card[i], current_deck[i], similarities[i], cards.loc[current_deck[i]]['edhrecRank']))

        # Sort rules based on EDHRec
        rules_2 = sorted(rules, key=lambda x:x[3])

        # Using rules of size 2, generate rules of size 3 
        # and calculate support (similarity) and confidence
        rules = []

        # Similar to rules_2 generation, find maximally similar rule
        # that is distinct from input card
        for card in current_deck:
            max_similarity = -1
            max_index_a = sys.maxsize
            max_index_b = sys.maxsize
            for rule in rules_2:	

                # Get max similarity of card to either card in rule
                card_similarity = self.get_similarity(rule[0], card)
                if card_similarity > max_similarity and rule[0] != card and rule[1] != card:
                    max_similarity = card_similarity 
                    max_index_a = rule[0]
                    max_index_b = rule[1]
                    rule_2_similarity = rule[2]

            # Find denominator for support by getting 
            # number of occurrences of first card 
            # in list
            denominator = 0
            for compared_card in sim_card:
                if compared_card == max_index_a:
                    denominator += 1

            # Take sum of rule similarity and max / denominator
            support_3 = (card_similarity + rule_2_similarity) / denominator

            # Take sum of rule similarity and max / similarity 
            # to compute confidence for rules of size 3
            confidence = ((card_similarity + rule_2_similarity) / rule_2_similarity) / len(self.rule_phrases[max_index_a])
            rules.append((max_index_a, max_index_b, card, support_3, confidence, cards.loc[card]['edhrecRank']))

        # Sort rules based on confidence
        rules_3 = sorted(rules, key=lambda x:x[4], reverse=True)

        # Print top ten confidence scores
        print("Top Ten Confidence Scores:")
        for rule in rules_3[:10]:
            print('Cards: {}, {}, {} Support: {} Confidence: {}'.format(cards.loc[rule[0]]['name'], cards.loc[rule[1]]['name'], cards.loc[rule[2]]['name'], rule[3], rule[4]))

        # Return all rules generated
        return rules_2, rules_3
    
    # Return related cards using given Commander
    def get_related_cards(self, commander_name, n_clusters = 6):

        # Filter for color identity and print number of legal cards
        comm_cards = self.filter_identity(commander_name)
        self.df_related_commander_cards = comm_cards
        print("Total number of cards legal for deck: {}".format(len(self.df_related_commander_cards.index)))

        # Extract all rules phrases for Apriori comparison
        rule_phrases = {}
        for i in self.df_cards.index:
            rule_phrases[i] = self.extract_text_perms(self.df_cards.loc[i])
        self.rule_phrases = rule_phrases

        # Perform K-Means clustering for k clusters on cards
        clusters = self.k_means(n_clusters, comm_cards)
        # Print length of each cluster
        for i in range(len(clusters)):
            print("Number of cards in cluster {}: {}".format(i, len(clusters[i])))

        # Generate rules of length 2 for deck
        #rules_2 is tuple with card in deck, recommended card, similarity (support) score, EDHRec
        rules_2, rules_3 = self.apriori(commander_name, clusters, comm_cards)

        # Sorts rules 2 in descending order based on similarity (support) score
        rules_2 = sorted(rules_2, key = lambda x:x[2], reverse=True)

        # Slices data frame into related cards df and saves it to class
        rules_2 = np.array(rules_2)
        sim_card_indices = rules_2[:, 1]
        related_cards = self.df_current_cards.loc[sim_card_indices, :]

        # Adds sim score to the data frame
        related_cards['sim_score'] = rules_2[:, 2]

        # Saves related cards to class and drops Commander
        related_cards = related_cards.iloc[1:].reset_index(drop=True)
        self.df_related_cards = related_cards

        # Return related cards
        return related_cards
    
    # Trains and sets the linear regression model
    def train_linear_model(self):
        model = SimpleLinearRegression()

        # Returns EDHRec and similarity (support) scores 
        # of related cards and trains the model on them
        df_edhrec = np.array(self.df_related_cards['edhrecRank'].tolist())
        df_sim_scores = np.array(self.df_related_cards['sim_score'].tolist())
        model.fit(df_sim_scores, df_edhrec)

        self.linear_model = model
        return model

        
    # Computes the upcoming recommended cards
    # Also sets the upcoming cards to the class
    def compute_upcoming_recommendations(self, commander_name, max_cards = 10):

        # Returns similarity score of upcoming cards to commander
        current_deck = [self.df_cards[self.df_cards.name == commander_name].index[0]]
        sim_card = [self.df_cards[self.df_cards.name == commander_name].index[0]]
        similarities = [1]

        # Obtain rules for cards not in training data
        # but similar to Commander (Upcoming cards)
        while len(current_deck) < 63:

            # Find maximally similar card to display from test data
            for card_a in current_deck[:(len(current_deck) // 50) + 1]:
                max_similarity = -1
                max_index_a = sys.maxsize
                max_index_b = sys.maxsize
                for index, card_b in self.df_upcoming_cards.iterrows():
                    card_similarity = self.get_similarity(card_a, index)
                    if card_similarity > max_similarity and index not in current_deck:
                        max_similarity = card_similarity 
                        max_index_a = card_a
                        max_index_b = index
			
            # Break before appending last card
            if len(current_deck) > 63:
                break
            
            # Append rules to list
            sim_card.append(max_index_a)
            current_deck.append(max_index_b)
            similarities.append(max_similarity)

        # Create tuple of rules
        rules = []
        for i in range(len(current_deck)):
            rules.append((sim_card[i], current_deck[i], similarities[i], self.df_cards.loc[current_deck[i]]['edhrecRank']))

        # Sort rules based on similarity and removes Commander
        rules = sorted(rules, key = lambda x:x[2], reverse=True)
        rules = rules[1:]

        # Slices sorted indices
        upcoming_indices = np.array([x[1] for x in rules])

        # Trains linear model for edhrec prediction
        lnr_model = self.train_linear_model()

        # Adds associated card data in order to a list of tuples
        upcoming_cards_data = []
        # List structure: index, card name, colorIdentity, 
        # types, text, similarity score, edhrecActual, edhrecPrediction
        counter = 0
        for comm_index, card_index, sim_score, edhrec in rules:
            row = self.df_upcoming_cards.loc[card_index]
            predicted_edhrec = lnr_model.predict(sim_score)
            upcoming_cards_data.append((card_index, row['name'], row['manaCost'], row['types'], row['text'], sim_score, row['edhrecRank'], predicted_edhrec))
            counter = counter + 1
            if counter > max_cards:
                break
        self.upcoming_cards_data = upcoming_cards_data
        return upcoming_cards_data
    
    # Returns upcoming card names for GUI
    def get_upcoming_card_names(self):
        cardnames = [row[1] for row in self.upcoming_cards_data]
        return {'cardnames': cardnames}
    
    # Returns related card names for GUI
    def get_related_card_names(self):
        cardnames = self.df_related_cards['name'].tolist()
        return {'cardnames': cardnames}
    
    # Return data related to an upcoming card
    def get_upcoming_card_data(self, cardname):

        # Replace NaN with readable value
        updated_data = [
            tuple('manaCost not found' if (isinstance(x, float) and math.isnan(x)) else x for x in row)
            for row in self.upcoming_cards_data]
        df_cards = pd.DataFrame(updated_data)
        df_cards.columns = ['card_index', 'name', 'manaCost', 'types', 'text', 'sim_score', 'edhrecRank', 'edhrecRank_prediction']

        # Return entry corresponding to card name
        row = df_cards[df_cards['name'] == cardname].iloc[0]

        # Properly format card rules text for GUI display
        card_text = row['text']
        card_text = card_text.replace('\\n', '- ')

        # Store values in dict
        card = {
            'cardname': cardname,
            'types': row['types'],
            'text': card_text,
            'manaCost': row['manaCost'],
            'cardtext': card_text,
            'edhrec': row['edhrecRank'],

            'edhrec_predicted': row['edhrecRank_prediction'],
            'sim_score': row['sim_score']
            } 

        return card
        
# Main method for CLI testing, for GUI
# start Flask server with:
# python3 -m flask --app app run
mtg = MTGProject()
commander_name = 'Grey Knight Paragon'
commander_name = "Yuriko, the Tiger's Shadow"
upcoming_card_name = "Shroudstomper"
related_cards = mtg.get_related_cards(commander_name)
mtg.compute_upcoming_recommendations(commander_name)
mtg.get_upcoming_card_names()
t = mtg.get_related_card_names()
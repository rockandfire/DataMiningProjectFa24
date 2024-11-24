import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import json
import sys
import re
from typing import List, Dict, Set, Tuple
from collections import defaultdict


class MTGDeckBuilder:
    def __init__(self, cards_file: str = 'cards.csv', keywords_file: str = 'Keywords.json', debug: bool = True):
        # Initialize the deck builder with card data and keywords.
        self.debug = debug
        self.cards_df = self._load_and_clean_data(cards_file)
        self.keywords = self._load_keywords(keywords_file)
        self.regression_model = None
        if self.debug:
            print(f"Loaded {len(self.cards_df)} cards")

    def _load_and_clean_data(self, cards_file: str) -> pd.DataFrame:
        # Load and clean the card data.
        print("Loading and cleaning card data...")
        df = pd.read_csv(cards_file, low_memory=False)

        # Filter for paper cards and remove duplicates
        df = df[df.availability.str.contains('paper', na=False)].drop_duplicates(subset='name', keep='last')

        # Keep essential columns
        keep_cols = ['name', 'colorIdentity', 'keywords', 'text', 'edhrecRank',
                     'manaValue', 'uuid', 'availability', 'isOnlineOnly', 'isTextless']
        df = df[keep_cols].copy()

        # Handle NaN values
        df['edhrecRank'] = df['edhrecRank'].fillna(df['edhrecRank'].max())
        df['manaValue'] = df['manaValue'].fillna(0)
        df['text'] = df['text'].fillna('')
        df['keywords'] = df['keywords'].fillna('')

        if self.debug:
            print(f"Cleaned data shape: {df.shape}")

        return df

    def _load_keywords(self, keywords_file: str) -> List[str]:
        # Load MTG keywords from JSON file.
        with open(keywords_file) as f:
            return json.load(f)

    def _analyze_card_text(self, text: str) -> Dict[str, Set[str]]:
        # Break down card text into meaningful mechanical components.
        if not isinstance(text, str):
            return {'counters': set(), 'triggers': set(), 'effects': set()}

        # Clean up text
        text = text.lower().replace('\n', ' ')

        # Find all counter types
        counter_pattern = r"([+-]\d+/[+-]\d+|[a-z]+) counter"
        counters = set(re.findall(counter_pattern, text))

        # Find trigger conditions
        triggers = set()
        trigger_words = ['whenever', 'when', 'at']
        words = text.split()
        for i, word in enumerate(words):
            if word in trigger_words and i < len(words) - 3:
                triggers.add(' '.join(words[i:i + 4]))

        # Find effects
        effects = set()
        effect_words = ['put', 'add', 'place', 'remove', 'double', 'proliferate']
        for i, word in enumerate(words):
            if word in effect_words and i < len(words) - 3:
                effects.add(' '.join(words[i:i + 4]))

        return {
            'counters': counters,
            'triggers': triggers,
            'effects': effects
        }

    def _get_color_identity_valid_cards(self, commander_name: str = None,
                                        color_identity: List[str] = None) -> pd.DataFrame:
        # Get valid cards for color identity from either a commander name or color list.
        if commander_name:
            commander_identity = set(self.cards_df[
                                         self.cards_df.name == commander_name
                                         ]['colorIdentity'].iloc[0].split(', '))
        else:
            commander_identity = set(color_identity)

        if self.debug:
            print(f"Commander color identity: {commander_identity}")

        indices = []
        for row in self.cards_df.index:
            if type(self.cards_df.loc[row]['colorIdentity']) is float:
                identity = set()
            else:
                identity = set(self.cards_df.loc[row]['colorIdentity'].split(', '))
            if identity <= commander_identity:
                indices.append(row)

        valid_cards = self.cards_df[self.cards_df.index.isin(indices)]

        if self.debug:
            print(f"Found {len(valid_cards)} valid cards for color identity")

        return valid_cards

    def _manual_kmeans(self, cards: pd.DataFrame, k: int) -> List[pd.DataFrame]:
        # Implement manual k-means clustering using EDHRec rank.
        if self.debug:
            print(f"\nPerforming k-means clustering with k={k}")

        centroids = []

        # Initialize random centroids
        for i in range(k):
            centroids.append(np.random.randint(
                cards['edhrecRank'].min(),
                cards['edhrecRank'].max()
            ))

        if self.debug:
            print(f"Initial centroids: {centroids}")

        # Create empty clusters
        assignments = [[] for _ in range(k)]

        # Assign cards to nearest centroid
        for index in cards.index:
            min_distance = sys.maxsize
            assigned_centroid_index = centroids.index(max(centroids))

            for rank in centroids:
                if abs(cards.loc[index]['edhrecRank'] - rank) < min_distance:
                    assigned_centroid_index = centroids.index(rank)
                    min_distance = abs(cards.loc[index]['edhrecRank'] - rank)
            assignments[assigned_centroid_index].append(index)

        # Update centroids and create clusters
        clustered_cards = []
        for i in range(k):
            if len(assignments[i]) > 0:
                sum_rank = sum(cards.loc[index]['edhrecRank'] for index in assignments[i])
                centroids[i] = sum_rank / len(assignments[i])
                clustered_cards.append((centroids[i], assignments[i]))

        # Sort clusters by centroid value
        clustered_cards.sort(key=lambda x: x[0])

        # Convert to list of DataFrames
        df_clusters = [
            cards[cards.index.isin(cluster[1])]
            for cluster in clustered_cards
        ]

        if self.debug:
            print("\nFinal clusters:")
            for i, cluster in enumerate(df_clusters):
                print(f"Cluster {i + 1}: {len(cluster)} cards, mean rank: {cluster['edhrecRank'].mean():.2f}")

        return df_clusters

    def _calculate_card_similarity(self, commander_card: pd.Series, candidate_card: pd.Series) -> Tuple[float, Dict]:
        # Calculate similarity between commander and candidate card with detailed breakdown.
        # Get text analysis for both cards
        commander_analysis = self._analyze_card_text(commander_card['text'])
        candidate_analysis = self._analyze_card_text(candidate_card['text'])

        # Calculate similarities for each mechanic type
        similarities = {
            'counters': len(commander_analysis['counters'] & candidate_analysis['counters']) /
                        max(len(commander_analysis['counters']), 1),
            'triggers': len(commander_analysis['triggers'] & candidate_analysis['triggers']) /
                        max(len(commander_analysis['triggers']), 1),
            'effects': len(commander_analysis['effects'] & candidate_analysis['effects']) /
                       max(len(commander_analysis['effects']), 1)
        }

        # Calculate mana and rank factors
        mana_factor = 1.0 - (candidate_card['manaValue'] / 15)
        rank_factor = 1.0 - (candidate_card['edhrecRank'] / self.cards_df['edhrecRank'].max())

        # Weights for mechanics
        mechanic_weights = {
            'counters': 0.4,
            'triggers': 0.3,
            'effects': 0.3
        }

        # Calculate mechanics score
        mechanics_score = sum(mechanic_weights[k] * similarities[k] for k in mechanic_weights)

        # Final score weights
        final_weights = {
            'mechanics': 0.7,
            'mana_cost': 0.2,
            'rank': 0.1
        }

        # Calculate final score
        total_similarity = (
                final_weights['mechanics'] * mechanics_score +
                final_weights['mana_cost'] * mana_factor +
                final_weights['rank'] * rank_factor
        )

        # Create detailed breakdown
        details = {
            'shared_counters': commander_analysis['counters'] & candidate_analysis['counters'],
            'shared_triggers': commander_analysis['triggers'] & candidate_analysis['triggers'],
            'shared_effects': commander_analysis['effects'] & candidate_analysis['effects'],
            'similarity_breakdown': {
                'mechanics_score': mechanics_score,
                'mana_factor': mana_factor,
                'rank_factor': rank_factor,
                'total': total_similarity
            }
        }

        return total_similarity, details

    def _find_associated_cards(self, commander_card: pd.Series, valid_cards: pd.DataFrame,
                               threshold: float = 0.2, max_cards: int = 63) -> pd.DataFrame:
        # Find associated cards with detailed similarity analysis.
        if self.debug:
            print(f"\nAnalyzing commander: {commander_card['name']}")
            commander_analysis = self._analyze_card_text(commander_card['text'])
            print("Commander mechanics:")
            for mech_type, mechs in commander_analysis.items():
                if mechs:
                    print(f"{mech_type.capitalize()}: {mechs}")

        card_scores = []

        for _, card in valid_cards.iterrows():
            if card['name'] != commander_card['name']:
                similarity, details = self._calculate_card_similarity(commander_card, card)
                if similarity >= threshold:
                    card_scores.append((card, similarity, details))

        # Sort by similarity score
        card_scores.sort(key=lambda x: x[1], reverse=True)

        # Convert to DataFrame
        if card_scores:
            recommended_cards = pd.DataFrame([score[0] for score in card_scores[:max_cards]])
            recommended_cards['similarity_score'] = [score[1] for score in card_scores[:max_cards]]
            recommended_cards['similarity_details'] = [score[2] for score in card_scores[:max_cards]]

            if self.debug:
                print(f"\nTop 5 recommended cards with details:")
                for _, row in recommended_cards.head().iterrows():
                    print(f"\n{row['name']} (Score: {row['similarity_score']:.3f})")
                    details = row['similarity_details']
                    print(f"Shared mechanics:")
                    print(f"- Counters: {details['shared_counters']}")
                    print(f"- Triggers: {details['shared_triggers']}")
                    print(f"- Effects: {details['shared_effects']}")
        else:
            recommended_cards = pd.DataFrame()

        return recommended_cards

    def _prepare_regression_features(self, cards: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        # Prepare features for EDHRec ranking prediction.
        if self.debug:
            print("\nPreparing regression features...")

        # Convert text to numeric features using mechanics analysis
        card_features = []
        for _, card in cards.iterrows():
            mechanics = self._analyze_card_text(card['text'])
            features = {
                'manaValue': card['manaValue'],
                'num_counters': len(mechanics['counters']),
                'num_triggers': len(mechanics['triggers']),
                'num_effects': len(mechanics['effects'])
            }
            card_features.append(features)

        # Convert to DataFrame
        X = pd.DataFrame(card_features)
        y = cards['edhrecRank']

        if self.debug:
            print(f"Feature matrix shape: {X.shape}")
            print("Features used:", list(X.columns))

        return X, y

    def _train_regression_model(self, X: np.ndarray, y: np.ndarray) -> Tuple[LinearRegression, Dict]:
        # Train linear regression model for EDHRec ranking prediction.
        if self.debug:
            print("\nTraining regression model...")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Calculate performance metrics
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)

        metrics = {
            'train_mse': np.mean((y_train - train_preds) ** 2),
            'test_mse': np.mean((y_test - test_preds) ** 2),
            'feature_importance': dict(zip(X.columns, model.coef_))
        }

        if self.debug:
            print("\nModel performance:")
            print(f"Train MSE: {metrics['train_mse']:.2f}")
            print(f"Test MSE: {metrics['test_mse']:.2f}")
            print("\nFeature importance:")
            for feature, importance in metrics['feature_importance'].items():
                print(f"{feature}: {importance:.2f}")

        return model, metrics

    def analyze_deck(self, commander_name: str, k_clusters: int = 4) -> Dict:
        # Analyze an existing commander.
        if commander_name not in self.cards_df['name'].values:
            raise ValueError(f"Commander '{commander_name}' not found in database")

        if self.debug:
            print(f"\nStarting analysis for commander: {commander_name}")

        # Get commander card and valid cards
        commander_card = self.cards_df[self.cards_df['name'] == commander_name].iloc[0]
        valid_cards = self._get_color_identity_valid_cards(commander_name=commander_name)

        # Cluster cards
        clusters = self._manual_kmeans(valid_cards, k_clusters)

        # Find associated cards
        associated_cards = self._find_associated_cards(commander_card, valid_cards)

        # Train regression model
        X, y = self._prepare_regression_features(associated_cards)
        self.regression_model, regression_metrics = self._train_regression_model(X, y)

        # Prepare results
        results = {
            'commander': {
                'card': commander_card.to_dict(),
                'analysis': self._analyze_card_text(commander_card['text'])
            },
            'clusters': [
                {
                    'centroid_rank': cluster['edhrecRank'].mean(),
                    'size': len(cluster),
                    'cards': cluster.to_dict('records')
                }
                for cluster in clusters
            ],
            'recommended_cards': associated_cards.to_dict('records') if not associated_cards.empty else [],
            'regression_analysis': {
                'metrics': regression_metrics,
                'feature_importance': regression_metrics['feature_importance']
            }
        }

        return results


    def analyze_future_card(self, card_name: str, card_text: str, mana_value: float,
                            color_identity: List[str], k_clusters: int = 4) -> Dict:
        # Analyze a future/unreleased card.
        if self.debug:
            print(f"\nAnalyzing future card: {card_name}")

        # Create temporary card entry
        future_card = pd.Series({
            'name': card_name,
            'text': card_text,
            'manaValue': mana_value,
            'colorIdentity': ', '.join(color_identity),
            'edhrecRank': -1,  # placeholder
            'keywords': '',
            'uuid': 'future_card',
            'availability': 'unreleased',
            'isOnlineOnly': False,
            'isTextless': False
        })

        # Get valid cards for color identity
        valid_cards = self._get_color_identity_valid_cards(color_identity=color_identity)

        # Analyze mechanics
        mechanics = self._analyze_card_text(card_text)
        if self.debug:
            print("\nMechanics found:")
            for mech_type, mechs in mechanics.items():
                if mechs:
                    print(f"{mech_type.capitalize()}: {mechs}")

        # Find associated cards
        associated_cards = self._find_associated_cards(future_card, valid_cards)

        # Predict EDHRec ranking
        features = pd.DataFrame([{
            'manaValue': mana_value,
            'num_counters': len(mechanics['counters']),
            'num_triggers': len(mechanics['triggers']),
            'num_effects': len(mechanics['effects'])
        }])

        if self.regression_model is None:
            # Train model if not already trained
            X, y = self._prepare_regression_features(associated_cards)
            self.regression_model, regression_metrics = self._train_regression_model(X, y)

        predicted_rank = self.regression_model.predict(features)[0]

        # Cluster recommended cards
        clusters = self._manual_kmeans(associated_cards, k_clusters)

        return {
            'card_analysis': {
                'name': card_name,
                'mechanics': mechanics,
                'predicted_rank': predicted_rank
            },
            'clusters': [
                {
                    'centroid_rank': cluster['edhrecRank'].mean(),
                    'size': len(cluster),
                    'cards': cluster.to_dict('records')
                }
                for cluster in clusters
            ],
            'recommended_cards': associated_cards.to_dict('records') if not associated_cards.empty else []
        }

    def _is_valid_commander(self, card: pd.Series) -> bool:
        # Check if a card is a valid commander.
        if pd.isna(card['text']):
            return False

        text = card['text'].lower()
        return any([
            'can be your commander' in text,
            'commander' in text and 'creature' in text,
            'legendary creature' in text,
            # Add specific cases we know about
            card['name'] in [
                'Shalai and Hallar',
                'Shalai, Voice of Plenty',
                # Add other known commanders that might be missed
            ]
        ])

    def get_valid_commanders(self) -> pd.Series:
        # Get all valid commanders in the database.
        return self.cards_df[self.cards_df.apply(self._is_valid_commander, axis=1)]['name'].sort_values()


def main():
    # Interactive interface for the deck builder.
    deck_builder = MTGDeckBuilder(debug=True)

    while True:
        print("\nMTG Deck Builder")
        print("1. Analyze existing commander")
        print("2. Analyze future card")
        print("3. Exit")

        choice = input("\nEnter your choice (1-3): ")

        if choice == '1':
            # Show available commanders
            print("\nLoading commander options...")
            commanders = deck_builder.get_valid_commanders()

            print(f"\nFound {len(commanders)} potential commanders.")
            print("\nSample of available commanders:")
            print(commanders.head(10).to_list())

            commander_name = input("\nEnter commander name (or part of name to search): ")
            matches = commanders[commanders.str.contains(commander_name, case=False)]

            if len(matches) == 0:
                print("No commanders found matching that name.")
                continue
            elif len(matches) > 1:
                print("\nMultiple commanders found:")
                for i, name in enumerate(matches, 1):
                    print(f"{i}. {name}")
                try:
                    choice = int(input("\nEnter the number of your choice: ")) - 1
                    commander_name = matches.iloc[choice]
                except (ValueError, IndexError):
                    print("Invalid choice.")
                    continue
            else:
                commander_name = matches.iloc[0]

            try:
                results = deck_builder.analyze_deck(commander_name)
                _display_results(results)
            except Exception as e:
                print(f"Error analyzing deck: {e}")

        elif choice == '2':
            print("\nAnalyze Future Card")
            card_name = input("Enter card name: ")
            card_text = input("Enter card text: ")
            mana_value = float(input("Enter mana value: "))
            print("\nEnter color identity (space-separated):")
            print("Use W for White, U for Blue, B for Black, R for Red, G for Green")
            color_identity = input("Colors: ").upper().split()

            try:
                results = deck_builder.analyze_future_card(
                    card_name=card_name,
                    card_text=card_text,
                    mana_value=mana_value,
                    color_identity=color_identity
                )
                _display_results(results, is_future_card=True)
            except Exception as e:
                print(f"Error analyzing future card: {e}")

        elif choice == '3':
            print("\nGoodbye!")
            break
        else:
            print("\nInvalid choice. Please try again.")


def _display_results(results: Dict, is_future_card: bool = False):
    # Display analysis results.
    if is_future_card:
        print(f"\n=== Analysis for Future Card: {results['card_analysis']['name']} ===")
        print("\nMechanics Found:")
        for mech_type, mechs in results['card_analysis']['mechanics'].items():
            if mechs:
                print(f"{mech_type.capitalize()}: {mechs}")
        print(f"\nPredicted EDHRec Rank: {results['card_analysis']['predicted_rank']:.2f}")
    else:
        print(f"\n=== Analysis for {results['commander']['card']['name']} ===")
        print("\nCommander Mechanics:")
        for mech_type, mechs in results['commander']['analysis'].items():
            if mechs:
                print(f"{mech_type.capitalize()}: {mechs}")

    print("\n=== Cluster Analysis ===")
    for i, cluster in enumerate(results['clusters'], 1):
        print(f"\nCluster {i}:")
        print(f"Mean EDHRec Rank: {cluster['centroid_rank']:.2f}")
        print(f"Number of cards: {cluster['size']}")

    print("\n=== Top Card Recommendations ===")
    for card in results['recommended_cards'][:10]:
        print(f"\n{card['name']}")
        print(f"EDHRec Rank: {card['edhrecRank']}")
        if 'similarity_score' in card:
            print(f"Similarity Score: {card['similarity_score']}")
            if 'similarity_details' in card:
                details = card['similarity_details']
                if details['shared_counters']:
                    print(f"Shared Counters: {details['shared_counters']}")
                if details['shared_triggers']:
                    print(f"Shared Triggers: {details['shared_triggers']}")
                if details['shared_effects']:
                    print(f"Shared Effects: {details['shared_effects']}")


if __name__ == "__main__":
    main()

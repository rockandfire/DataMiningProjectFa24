// src/App.jsx
import React, { useState, useEffect } from 'react';

const API_URL = 'http://localhost:5000/api';

const CardDetailsPanel = ({ card }) => {
  if (!card) return null;

  return (
    <div className="bg-white dark:bg-slate-800 p-6 rounded-lg shadow-lg">
      <h3 className="text-2xl font-semibold mb-4 text-slate-800 dark:text-white">{card.name}</h3>
      <div className="space-y-4">
        <div>
          <p className="font-medium text-slate-800 dark:text-white">EDHRec Rank:</p>
          <p className="text-slate-800 dark:text-white">{card.edhrecRank?.toFixed(0)}</p>
        </div>
        
        {card.similarity_score && (
          <div>
            <p className="font-medium text-slate-800 dark:text-white">Similarity Score:</p>
            <p className="text-slate-800 dark:text-white">{card.similarity_score.toFixed(3)}</p>
          </div>
        )}

        <div>
          <p className="font-medium text-slate-800 dark:text-white">Mana Value:</p>
          <p className="text-slate-800 dark:text-white">{card.manaValue}</p>
        </div>

        <div>
          <p className="font-medium text-slate-800 dark:text-white">Card Text:</p>
          <p className="whitespace-pre-wrap text-slate-800 dark:text-white">{card.text || 'No card text available'}</p>
        </div>

        <div>
          <p className="font-medium text-slate-800 dark:text-white">Keywords:</p>
          <p className="text-slate-800 dark:text-white">{card.keywords || 'No keywords'}</p>
        </div>

        {card.similarity_details && (
          <div>
            <p className="font-medium text-slate-800 dark:text-white">Shared Mechanics:</p>
            {card.similarity_details.shared_counters?.size > 0 && (
              <div>
                <p className="font-medium text-sm text-slate-800 dark:text-white">Counters:</p>
                <p className="text-slate-800 dark:text-white">{Array.from(card.similarity_details.shared_counters).join(', ')}</p>
              </div>
            )}
            {card.similarity_details.shared_triggers?.size > 0 && (
              <div>
                <p className="font-medium text-sm text-slate-800 dark:text-white">Triggers:</p>
                <p className="text-slate-800 dark:text-white">{Array.from(card.similarity_details.shared_triggers).join(', ')}</p>
              </div>
            )}
            {card.similarity_details.shared_effects?.size > 0 && (
              <div>
                <p className="font-medium text-sm text-slate-800 dark:text-white">Effects:</p>
                <p className="text-slate-800 dark:text-white">{Array.from(card.similarity_details.shared_effects).join(', ')}</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

const MTGDeckBuilder = () => {
  const [currentView, setCurrentView] = useState('home');
  const [commanderSearch, setCommanderSearch] = useState('');
  const [commanders, setCommanders] = useState([]);
  const [selectedCommander, setSelectedCommander] = useState(null);
  const [commanderResults, setCommanderResults] = useState(null);
  const [selectedCard, setSelectedCard] = useState(null);
  const [futureCardForm, setFutureCardForm] = useState({
    cardName: '',
    manaValue: '',
    colorIdentity: [],
    cardText: ''
  });
  const [futureCardResults, setFutureCardResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (commanderSearch.length >= 2) {
      fetch(`${API_URL}/commanders?search=${encodeURIComponent(commanderSearch)}`)
        .then(res => res.json())
        .then(data => {
          console.log('Found commanders:', data);
          setCommanders(data);
        })
        .catch(error => {
          console.error('Error fetching commanders:', error);
          setError('Failed to fetch commanders');
        });
    } else {
      setCommanders([]);
    }
  }, [commanderSearch]);

  const analyzeCommander = async () => {
    if (!selectedCommander) return;
    
    setIsLoading(true);
    setError(null);
    try {
      console.log('Analyzing commander:', selectedCommander);
      const response = await fetch(`${API_URL}/analyze-commander`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ commander: selectedCommander })
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      console.log('Commander analysis results:', data);
      setCommanderResults(data);
    } catch (error) {
      console.error('Error analyzing commander:', error);
      setError(error.message);
    } finally {
      setIsLoading(false);
    }
  };

  const analyzeFutureCard = async () => {
    setIsLoading(true);
    setError(null);
    try {
      console.log('Analyzing future card:', futureCardForm);
      const response = await fetch(`${API_URL}/analyze-future-card`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(futureCardForm)
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      console.log('Future card analysis results:', data);
      setFutureCardResults(data);
    } catch (error) {
      console.error('Error analyzing future card:', error);
      setError(error.message);
    } finally {
      setIsLoading(false);
    }
  };

  const renderHome = () => (
    <div className="min-h-screen bg-slate-100 flex items-center justify-center">
      <div className="text-center p-8">
        <h1 className="text-4xl font-bold text-slate-800 mb-8">MTG Deck Builder Assistant</h1>
        <div className="space-y-4 md:space-y-0 md:space-x-4">
          <button
            onClick={() => setCurrentView('commander')}
            className="px-6 py-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
          >
            Analyze Existing Commander
          </button>
          <button
            onClick={() => setCurrentView('future')}
            className="px-6 py-3 bg-purple-500 text-white rounded-lg hover:bg-purple-600 transition-colors"
          >
            Analyze Future Card
          </button>
        </div>
      </div>
    </div>
  );
  
  // Update the renderAnalysisResults function to include card selection:
  const renderAnalysisResults = (results) => {
    if (!results) return null;
    
    return (
      <div className="space-y-6">
        {/* Card Analysis Section */}
        {results.card_analysis && (
            <div className="bg-white dark:bg-slate-800 p-6 rounded-lg shadow-lg">
            <h2 className="text-2xl font-semibold mb-4 text-slate-800 dark:text-white">Card Analysis</h2>
            <div className="space-y-4">
              <p className="font-medium text-slate-800 dark:text-white">Name: {results.card_analysis.name}</p>
              <p className="text-slate-800 dark:text-white">Predicted Rank: {results.card_analysis.predicted_rank?.toFixed(2)}</p>
              {results.card_analysis.mechanics && (
                <div>
                  <p className="font-medium text-slate-800 dark:text-white">Mechanics:</p>
                  {Object.entries(results.card_analysis.mechanics).map(([type, items]) => (
                    <div key={type} className="ml-4">
                      <p className="capitalize text-slate-800 dark:text-white">{type}: {Array.from(items).join(', ')}</p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}
        
        {/* Clusters Section */}
        {results.clusters && results.clusters.length > 0 && (
          <div className="bg-white dark:bg-slate-800 p-6 rounded-lg shadow-lg">
          <h2 className="text-2xl font-semibold mb-4 text-slate-800 dark:text-white">Clusters</h2>
          <div className="space-y-4">
            {results.clusters.map((cluster, index) => (
              <div key={index} className="p-4 bg-slate-50 dark:bg-slate-700 rounded-lg">
                <p className="font-medium text-slate-800 dark:text-white">Cluster {index + 1}</p>
                <p className="text-slate-800 dark:text-white">Mean Rank: {cluster.centroid_rank?.toFixed(2)}</p>
                <p className="text-slate-800 dark:text-white">Size: {cluster.size}</p>
                <div className="mt-2">
                  {cluster.cards.slice(0, 5).map((card, cardIndex) => (
                    <div
                      key={cardIndex}
                      className="ml-4 text-sm cursor-pointer text-slate-800 dark:text-white hover:text-blue-500"
                      onClick={() => setSelectedCard(card)}
                    >
                      {card.name} (Rank: {card.edhrecRank?.toFixed(0)})
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
        )}
        
        {/* Recommended Cards Section */}
        {results.recommended_cards && results.recommended_cards.length > 0 && (
          <div className="bg-white dark:bg-slate-800 p-6 rounded-lg shadow-lg">
            <h2 className="text-2xl font-semibold mb-4 text-slate-800 dark:text-white">Recommended Cards</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                {results.recommended_cards.slice(0, 10).map((card, index) => (
                  <div
                    key={index}
                    className={`p-4 rounded-lg cursor-pointer transition-colors
                      ${selectedCard === card 
                        ? 'bg-blue-100 dark:bg-blue-900 text-slate-800 dark:text-white' 
                        : 'bg-slate-50 dark:bg-slate-700 text-slate-800 dark:text-white hover:bg-slate-100 dark:hover:bg-slate-600'}`}
                    onClick={() => setSelectedCard(card)}
                  >
                    <p className="font-medium">{card.name}</p>
                    <div className="text-sm">
                      <p>Rank: {card.edhrecRank?.toFixed(0)}</p>
                      {card.similarity_score && (
                        <p>Similarity: {card.similarity_score.toFixed(3)}</p>
                      )}
                    </div>
                  </div>
                ))}
              </div>
              <div>
                <CardDetailsPanel card={selectedCard} />
              </div>
            </div>
          </div>
        )}
      </div>
    );
  };

  const renderCommanderAnalysis = () => (
    <div className="min-h-screen bg-slate-100 p-6">
      <div className="max-w-6xl mx-auto">
        <button 
          onClick={() => setCurrentView('home')}
          className="mb-6 text-blue-500 hover:text-blue-600 flex items-center gap-2"
        >
          ← Back to Home
        </button>
        
        {error && (
          <div className="mb-6 bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
            {error}
          </div>
        )}
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-white p-6 rounded-lg shadow-lg">
            <h2 className="text-2xl font-semibold mb-4">Search Commander</h2>
            <input
              type="text"
              value={commanderSearch}
              onChange={(e) => setCommanderSearch(e.target.value)}
              placeholder="Enter commander name (minimum 2 characters)..."
              className="w-full px-4 py-2 border rounded-lg mb-4"
            />
            <div className="h-64 border rounded-lg p-2 mb-4 overflow-y-auto">
              {commanders.map((commander, index) => (
                <div
                  key={index}
                  onClick={() => setSelectedCommander(commander)}
                  className={`p-2 cursor-pointer hover:bg-slate-100 
                    ${selectedCommander === commander ? 'bg-blue-100' : ''}`}
                >
                  {commander}
                </div>
              ))}
            </div>
            <button 
              onClick={analyzeCommander}
              className="w-full py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors disabled:opacity-50"
              disabled={!selectedCommander || isLoading}
            >
              {isLoading ? 'Analyzing...' : 'Analyze Commander'}
            </button>
          </div>

          {renderAnalysisResults(commanderResults)}
        </div>
      </div>
    </div>
  );

  const renderFutureCardAnalysis = () => (
    <div className="min-h-screen bg-slate-100 p-6">
      <div className="max-w-6xl mx-auto">
        <button 
          onClick={() => setCurrentView('home')}
          className="mb-6 text-purple-500 hover:text-purple-600 flex items-center gap-2"
        >
          ← Back to Home
        </button>
        
        {error && (
          <div className="mb-6 bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
            {error}
          </div>
        )}
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-white p-6 rounded-lg shadow-lg">
            <h2 className="text-2xl font-semibold mb-4">Card Details</h2>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-1">Card Name</label>
                <input
                  type="text"
                  value={futureCardForm.cardName}
                  onChange={(e) => setFutureCardForm({
                    ...futureCardForm,
                    cardName: e.target.value
                  })}
                  className="w-full px-4 py-2 border rounded-lg"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium mb-1">Mana Value</label>
                <input
                  type="number"
                  value={futureCardForm.manaValue}
                  onChange={(e) => setFutureCardForm({
                    ...futureCardForm,
                    manaValue: e.target.value
                  })}
                  className="w-full px-4 py-2 border rounded-lg"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium mb-1">Color Identity</label>
                <div className="flex gap-4">
                  {['W', 'U', 'B', 'R', 'G'].map(color => (
                    <label key={color} className="flex items-center gap-1">
                      <input
                        type="checkbox"
                        checked={futureCardForm.colorIdentity.includes(color)}
                        onChange={(e) => {
                          const newColors = e.target.checked
                            ? [...futureCardForm.colorIdentity, color]
                            : futureCardForm.colorIdentity.filter(c => c !== color);
                          setFutureCardForm({
                            ...futureCardForm,
                            colorIdentity: newColors
                          });
                        }}
                        className="rounded"
                      />
                      {color}
                    </label>
                  ))}
                </div>
              </div>
              
              <div>
                <label className="block text-sm font-medium mb-1">Card Text</label>
                <textarea
                  value={futureCardForm.cardText}
                  onChange={(e) => setFutureCardForm({
                    ...futureCardForm,
                    cardText: e.target.value
                  })}
                  className="w-full px-4 py-2 border rounded-lg h-32"
                  placeholder="Enter card text..."
                />
              </div>
              
              <button
                onClick={analyzeFutureCard}
                className="w-full py-2 bg-purple-500 text-white rounded-lg hover:bg-purple-600 transition-colors disabled:opacity-50"
                disabled={isLoading || !futureCardForm.cardName}
              >
                {isLoading ? 'Analyzing...' : 'Analyze Card'}
              </button>
            </div>
          </div>

          {renderAnalysisResults(futureCardResults)}
        </div>
      </div>
    </div>
  );

  return (
    <div>
      {currentView === 'home' && renderHome()}
      {currentView === 'commander' && renderCommanderAnalysis()}
      {currentView === 'future' && renderFutureCardAnalysis()}
    </div>
  );
};

export default MTGDeckBuilder;
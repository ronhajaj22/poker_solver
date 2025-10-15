import React, { useState, useEffect, useRef } from 'react';
import GameSetup from './GameSetup';
import PokerTable from './PokerTable';
import './css_files/App.css';
import { setSelectionRange } from '@testing-library/user-event/dist/utils';

// Main App component controls the game flow
function App() {
  // State for game setup
  const [gameStarted, setGameStarted] = useState(false);
  const [players, setPlayers] = useState([]);
  const [mainPlayerIndex, setMainPlayerIndex] = useState(0);
  const [communityCards, setCommunityCards] = useState([]);
  const [pot, setPot] = useState(0);
  const [lastAction, setLastAction] = useState(null);
  const [players_to_act, setPlayersToAct] = useState([]);
  const [stage, setStage] = useState(0)
  const [isMainPlayerActive, setIsMainPlayerActive] = useState(false)
  const [bbSize, setBbSize] = useState(1)
  const [lastBetSize, setLastBetSize] = useState(bbSize);
  const [winners, setWinners] = useState([]);
  const [highlightedCards, setHighlightedCards] = useState([]);
  const [checkAction, setCheckAction] = useState('');
  const [isGamePaused, setIsGamePaused] = useState(false);
  const [isTooltipVisible, setIsTooltipVisible] = useState(false);
  const tooltipVisibleRef = useRef(false);
  // Watch for tooltip visibility changes
  useEffect(() => {
    tooltipVisibleRef.current = isTooltipVisible;
  }, [isTooltipVisible]);

  // Helper function to convert backend player format to frontend format
  const convertPlayerFormat = (backendPlayer, index) => ({
    id: index,
    position: backendPlayer.position,
    position_index: backendPlayer.position_index,
    name: backendPlayer.name,
    is_main_player: backendPlayer.is_main_player,
    stack_size: backendPlayer.stack_size,
    chips_in_pot: backendPlayer.chips_in_pot,
    is_folded: backendPlayer.is_folded || false,
    is_need_to_act: backendPlayer.is_need_to_act || false,
    cards: backendPlayer.hand || [] // Backend returns 'hand', frontend expects 'cards'
  });

  const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

  // Clear check action message
  const clearCheckAction = () => {
    setCheckAction('');
    setIsGamePaused(false);
  };

  // Resume game when user clicks anywhere
  const resumeGame = () => {
    if (isGamePaused) {
      setIsGamePaused(false);
    }
  };

  // Start game cycle when players are set
  useEffect(() => {
    if (players.length > 0 && !isGamePaused) {
      startGameCycle(stage);
    }
  }, [players.length, players_to_act, isGamePaused]);

  // Game cycle state
  
  // Start game handler: called from GameSetup
  const handleStartGame = async (numPlayers, stackSize) => {
    // Call backend to initialize game
    fetch('/api/game/new', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ num_players: numPlayers, stack_size: stackSize })
    })
      .then(res => res.json())
      .then(async data => {
        // Convert backend player format to frontend format
        let convertedPlayers = data.players.map((player, index) => 
          convertPlayerFormat(player, index)
        );        
        // Set up initial state from backend response
        let mPlayer = convertedPlayers.findIndex(p => p.is_main_player);
        setMainPlayerIndex(mPlayer);
        setPlayers(convertedPlayers);
        setCommunityCards([]);
        setPot(data.pot || 1.5); // Default pot size
        setGameStarted(true);
        setPlayersToAct([convertedPlayers[0].name]);
        setBbSize(bbSize);
      })
      .catch(error => {
        console.error('Error starting game:', error);
        alert('Failed to start game. Please try again.');
      });
  };

  const startGameCycle = async (stage) => {

    if (players_to_act == [] || players_to_act.length === 0) {
      if (players.filter(player => !player.is_folded).length === 1) {
        getWinner(players.filter(player => !player.is_folded))
      } else {
        if (stage === 3) {
          getWinner(players.filter(player => !player.is_folded))
          return;
        }
        deal_next_cards(stage)
        setStage(stage => stage + 1)
      }      
      return;
    }

    let player_to_act = players_to_act[0];
    if (player_to_act === players[mainPlayerIndex].name) {
      setIsMainPlayerActive(true)
      return;
    } else {
      await sleep(1000); // 1 second delay
      getPlayerActionFromServer(player_to_act);
    }
  }

  const getWinner = () => {  
    fetch('/api/game/show_down', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 'players': players })
    }).then(res => res.json())
    .then(async data => {
      finishHandWithWinner(data.winners, data.winning_hands)
    })
    .catch(error => {
      console.error('Error finding a winner', error);
      alert('Failed to find a winner.');
    });
  }

  const finishHandWithWinner = async (winners, winning_hands) => {     
    // Set the winner for chip animation
    await sleep(1000);
    
    if (winning_hands != [] && winning_hands != null) {
      let highlightedCards = createHighlightedCards(winning_hands);
      setHighlightedCards(highlightedCards); // highligted card are grayed out
    }

    // Update the winner's state to show WIN label and update stack
    console.log('Winners array:', winners);
    setPlayers(prevPlayers => {
      const updated = prevPlayers.map(player => {
        if (winners.includes(player.name)) {
          console.log(`Setting WIN for player: ${player.name}`);
          return { 
            ...player, 
            showActionLabel: true, 
            lastAction: 'WIN',
            is_winner: true
          };
        }
        return player;
      });
      console.log('Updated players:', updated);
      return updated;
    });

    setWinners(winners);
    await sleep(1000);    
    
    // Hide the WIN label after 3 seconds and start new hand
    setTimeout(() => {
      setPlayers(prevPlayers => {
        const updated = prevPlayers.map(player => {
          if (winners.includes(player.name)) {
            return { ...player, showActionLabel: false, is_winner: false };
          }
          return player;
        });
        return updated;
      });
      setWinners([]); // Clear winner state
      setHighlightedCards([]); // Clear highlighted cards
      startNewHand();
    }, 3000);
  }

  const startNewHand = () => {
    fetch('/api/game/start_new_hand', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({})
    }).then(res => res.json())
    .then(async data => {
      let convertedPlayers = data.players.map((player, index) => 
        convertPlayerFormat(player, index)
      );
      convertedPlayers.sort((a, b) => a.position_index - b.position_index);
      let mPlayer = convertedPlayers.findIndex(p => p.is_main_player);
      setMainPlayerIndex(mPlayer);
      setPlayers(convertedPlayers);
      setPlayersToAct([convertedPlayers[0].name]);
      setPot(data.pot_size || 1.5); // Default pot size
      setLastBetSize(bbSize);
      setStage(0)
      setCommunityCards([])
    })
    .catch(error => {
      console.error('Error starting new hand', error);
      alert('Failed to start new hand. Please try again.');
    });
  }
      
  const handleMainPlayerAction = (action, betSize) => {
    setIsMainPlayerActive(false)
    if (action === 'CALL') {
      betSize = lastBetSize;
    }
    console.log("betSize: ", betSize)
    fetch('/api/game/send_main_player_action', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 'action': action, 'bet_size': parseFloat(betSize) })
    }).then(res => res.json())
      .then(data => {
        if (stage === 0) {
          setIsGamePaused(true); // change here to return to game
          // Store current tooltip state before timeout
          setTimeout(() => {
            if (!tooltipVisibleRef.current) {
              console.log("free the game no click")
              setIsGamePaused(false);
            }
          }, 3000);
        }
        
        // TODO - this is a feature, should be enabled only when the feature is enabled
        if (action === "FOLD" && players.filter(player => !player.is_folded).length > 2) {
          startNewHand();
        } else {
          handlePlayerAction(players[mainPlayerIndex].name, action, parseFloat(data.total_bet_size), parseFloat(data.added_amount), data.players_to_act, data.check_action);
        }
      })
      .catch(error => {
        console.error('Error sending main player action:', error);
      });
  }

  const getPlayerActionFromServer = (playerName) => {
    fetch('/api/game/player_action', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 'player': playerName })
    }).then(res => res.json())
      .then(data => {
        handlePlayerAction(playerName, data.action, parseFloat(data.total_bet_size), parseFloat(data.added_amount), data.players_to_act);
      })
      .catch(error => {
        console.error('Error sending player action:', error);
      });
  }

  const deal_next_cards = (stage) => {
    // Clear last bets before dealing next cards
    setLastBetSize(0);
    setPlayers(prevPlayers => prevPlayers.map(player => ({
      ...player,
      chips_in_pot: 0,
      raised_amount: 0
    })));
    
    fetch('/api/game/deal_next_cards', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ stage: stage })
    }).then(res => res.json())
      .then(data => {
        setLastBetSize(0);
        openCommunityCardsSetPlayersToAct(data.community_cards || [], data.players_to_act);
      })
      .catch(error => {
        console.error('Error sending player action:', error);
      });
  }

  const openCommunityCardsSetPlayersToAct = async (communityCards, players_to_act) => {
    let start = stage === 0 ? 0 : stage + 2;

    for (let i = start; i < communityCards.length; i++) {
      await sleep(500);
      setCommunityCards(prevCommunityCards => [...prevCommunityCards, communityCards[i]]);
    }
    setPlayersToAct(players_to_act)
  }
  
  // Function to update a player's state in the players array
  const handlePlayerAction = (playerName, action, totalBetSize = 0, addedAmount = 0, players_to_act = [], checkActionMessage = '') => {
    setPlayersToAct(players_to_act);
    // Move state updates outside the setPlayers callback
    if (action === 'RAISE' || action === 'CALL') {
      setPot(potSize => potSize + addedAmount);
      setLastBetSize(totalBetSize);
    }
    setLastAction(action);
    if (stage === 0) {
      setCheckAction(checkActionMessage);
    } else {
      setCheckAction('');
    }
    
    setPlayers(prevPlayers => {
      const updatedPlayers = prevPlayers.map(player => {
        if (player.name === playerName) {
          let updates = { showActionLabel: true, lastAction: action };
          if (action === 'FOLD') updates.is_folded = true;
          if (action === 'RAISE' || action === 'CALL') {
            updates.chips_in_pot = totalBetSize;
            updates.stack_size = player.stack_size - addedAmount;
          }
          return { ...player, ...updates };
        }
        return player;
      });

      return updatedPlayers;
    });

    // Hide the action label after 1 second
    setTimeout(() => {
      setPlayers(prevPlayers => prevPlayers.map(player => {
        if (player.name === playerName) {
          return { ...player, showActionLabel: false };
        }
        return player;
      }));
    }, action === 'FOLD' ? 1000 : 2000);
  };

  const createHighlightedCards = (winning_hands) => {
    // Collect all winning cards
    let highlighted_cards = [];
    for (let i = 0; i < winning_hands.length; i++) {
      for (let j = 0; j < winning_hands[i].hand.length; j++) {
        highlighted_cards.push({ rank: winning_hands[i].hand[j].rank, suit: winning_hands[i].hand[j].suit });
      }
    }
    
    return highlighted_cards;
  }

  // Render setup screen or poker table
  return (
    <div className="App">
      {!gameStarted ? (
        // Show setup screen
        <GameSetup onStartGame={handleStartGame} />
      ) : (
        // Show poker table
        <PokerTable
          players={players}
          mainPlayerIndex={mainPlayerIndex}
          communityCards={communityCards}
          pot={pot}
          mainPlayerActionHandler={handleMainPlayerAction}
          isMainPlayerActive={isMainPlayerActive}
          lastBetSize={lastBetSize}
          winners={winners}
          stage={stage}
          highlightedCards={highlightedCards}
          checkAction={checkAction}
          onClearCheckAction={clearCheckAction}
          isGamePaused={isGamePaused}
          onResumeGame={resumeGame}
          isTooltipVisible={isTooltipVisible}
          setIsTooltipVisible={setIsTooltipVisible}
          playersToAct={players_to_act}
        />
      )}
    </div>
  );
}

export default App; 



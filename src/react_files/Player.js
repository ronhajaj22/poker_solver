import React, { useState, useCallback, useEffect } from 'react';
import Card from './Card';
import ChipStack from './ChipStack';
import './css_files/Player.css';

// Helper function to format numbers - show decimals only if they exist
const formatNumber = (value) => {
  const num = parseFloat(value);
  if (isNaN(num)) return '0';
  // Round to 2 decimal places to handle floating point precision issues
  const rounded = Math.round(num * 100) / 100;
  return rounded % 1 === 0 ? rounded.toString() : rounded.toFixed(2);
};

// Player component renders a single player at the table
function Player({ player, isMain, isCardsHidden, isPlayerTurn, isNeedToAct, positionLabel, mainPlayerActionHandler, isMainPlayerActive, lastBetSize = 0, playerAngle = 0, stage = 0, pot = 0, highlightedCards = [], checkAction = '', onClearCheckAction, isGamePaused = false, onResumeGame, isTooltipVisible, setIsTooltipVisible }) {
  
  // Local state for card visibility
  const [localCardsHidden, setLocalCardsHidden] = useState(isCardsHidden);
  
  // Update local state when isCardsHidden prop changes
  useEffect(() => {
    setLocalCardsHidden(isCardsHidden);
  }, [isCardsHidden]);

  // Reset tooltip visibility when checkAction changes
  useEffect(() => {
    if (checkAction) {
      setIsTooltipVisible(false);
    }
  }, [checkAction]);

  // Hide tooltip when game resumes
  useEffect(() => {
    if (!isGamePaused) {
      setIsTooltipVisible(false);
    }
  }, [isGamePaused]);

  // Compute player CSS classes
  let playerClass = 'player';
  if (player.is_folded) playerClass += ' folded';
  if (isMain && isMainPlayerActive) playerClass += ' main-player';
  if (isPlayerTurn && !isMain) playerClass += ' need-to-act';
  if (player.is_winner) playerClass += ' winner';

  // Safely get cards array and filter out invalid cards
  const playerCards = (player.cards || []).filter(card => card && typeof card === 'object' && card.rank && card.suit);

  // Calculate chip position based on player angle
  // Returns where chips should be positioned relative to the player
  // Chips should appear "on the table" - between player and table center
  const getChipPosition = (angle) => {
    // Convert angle to degrees (0-360)
    let degrees = (angle * 180) / Math.PI;
    
    while (degrees < 0) degrees += 360;
    angle = degrees % 360 
    console.log("angle: ", angle);
    const ranges = [
      { from: 247.5, to: 292.5, pos: 'top' },
      {from: 292.5, to: 337.5, pos: 'top-left' },
      {from: 337.5, to: 360, pos: 'left' },
      {from: 0, to: 22.5, pos: 'left' },
      { from: 22.5, to: 67.5, pos: 'bottom-left' },
      { from: 67.5, to: 112.5, pos: 'bottom' },
      { from: 112.5, to: 157.5, pos: 'bottom-right' },
      { from: 157.5, to: 202.5, pos: 'right' },
      { from: 202.5, to: 247.5, pos: 'top-right' },
    ];
    return ranges.find(r => angle >= r.from && angle < r.to)?.pos;
  };

  // Render player UI
  return (
    <div className={playerClass} style={player.positionStyle}>
      {/* Action label above player box */}
      {player.showActionLabel && (
        <div className={`player-action-label ${player.lastAction === 'FOLD' ? 'fold-action' : ''} ${player.lastAction === 'WIN' ? 'win-action' : ''}`}>
          {player.lastAction === 'CHECK' && 'Check'}
          {player.lastAction === 'RAISE' && 'Bet '+ formatNumber(player.chips_in_pot)}
          {player.lastAction === 'CALL' && 'Call'}
          {player.lastAction === 'FOLD' && 'Fold'}
          {player.lastAction === 'WIN' && 'WIN!'}
        </div>
      )}
      
      {/* Question mark for check action feedback */}
      {checkAction && isMain && (
        <div className="check-action-question-mark" onClick={(e) => {
          e.stopPropagation(); // Prevent table click from triggering
          let wasVisible = isTooltipVisible;
          setIsTooltipVisible(!wasVisible);
          console.log("isTooltipVisible (player.js): ",  !wasVisible);
    
          // Resume the game when clicking the question mark
          if (wasVisible && onResumeGame) {
            console.log("wasVisible: ", wasVisible)
            onResumeGame();
          }
        }}>
          ?
          {isTooltipVisible && (
            <div className="check-action-tooltip">
              {checkAction}
            </div>
          )}
        </div>
      )}
      {/* Player position label (BTN, SB, BB, etc.) */}
      <div className="player-position">{positionLabel}</div>
      {/* Player name and stack */}
      <div className="player-info">
        <span className="player-name">{player.name}</span>
        <span className="player-stack">{formatNumber(player.stack_size)}BB</span>
      </div>

      {(!player.is_folded || player.lastAction === 'FOLD') && ( // TODO - change it later
          <div 
            className="player-cards" 
            onClick={() => setLocalCardsHidden(!localCardsHidden)}
            style={{ cursor: 'pointer' }}
          >
          {playerCards.length > 0 ? (
            playerCards.map((card, i) => (
              <Card 
                key={i} 
                card={card} 
                hidden={highlightedCards.length == 0 && localCardsHidden} 
                // actually it is - isGrayedOut
                isGrayedOut={highlightedCards.length > 0 && !highlightedCards.some(hc => hc.rank === card.rank && hc.suit === card.suit)}
              />
            ))
          ) : (
            [0, 1].map(i => (
              <div key={i} className="card card-back"></div>
            ))
          )}
        </div>
      )}
      {/* Show chips if player has raised */}
      {!player.is_folded && player.chips_in_pot > 0 && (
        <>
          <div className="player-chips">+{formatNumber(player.chips_in_pot)}</div>
          <ChipStack amount={player.chips_in_pot} position={getChipPosition(playerAngle)} />
        </>
      )}
    </div>
  );
}

export default Player; 
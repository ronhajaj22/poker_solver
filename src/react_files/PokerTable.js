import React, { useEffect, useState, useRef } from 'react';
import Player from './Player';
import Card from './Card';
import ChipStack from './ChipStack';
import ActionPanel from './ActionPanel';
import './css_files/PokerTable.css';

// Helper function to format numbers - show decimals only if they exist
const formatNumber = (value) => {
  const num = parseFloat(value);
  if (isNaN(num)) return '0';
  // Round to 2 decimal places to handle floating point precision issues
  const rounded = Math.round(num * 100) / 100;
  return rounded % 1 === 0 ? rounded.toString() : rounded.toFixed(2);
};

// Helper to arrange players so main player is always at the bottom (π/2), others evenly around the oval
function getPlayerPositionsOval(players, mainPlayerIndex) {
  const n = players.length;
  if (n === 0) return [];
  // Angle for main player (bottom center)
  const mainAngle = Math.PI * 3 / 2;
  // Angles for other players, skipping the bottom 
  const angleStep = (2 * Math.PI) / n;
  const positions = [];
  for (let i = 0; i < n; i++) {
    if (i === mainPlayerIndex) continue;
    // Distribute other players evenly, skipping the bottom
    const idx = i < mainPlayerIndex ? i - mainPlayerIndex + n : i - mainPlayerIndex;
    const angle = -(mainAngle + angleStep * idx) + Math.PI;
    positions.push({ player: players[i], angle });
  }
  // Add main player at the bottom
  positions.push({ player: players[mainPlayerIndex], angle: mainAngle});
  return positions;
}

// Helper to get absolute position on a perfectly centered oval
function getOvalPosition(angle, width, height) {
  // Center of oval (centered on the pot)
  const cx = width / 2;
  const cy = height / 2;
  // Radii: use 49% of width/height to push players to the very edge (with a tiny margin)
  const rx = width * 0.4;
  const ry = height * 0.37;
  //Calculate position, then subtract cx/cy to center in the container
  const x = cx + rx * Math.cos(angle) - cx;
  const y = cy - ry * Math.sin(angle) - cy ;
  return { left: `calc(50% + ${x}px)`, top: `calc(50% + ${y}px)` };
}

// PokerTable component renders the main game UI
function PokerTable({
  players,
  mainPlayerIndex,
  communityCards,
  pot,
  mainPlayerActionHandler,
  isMainPlayerActive,
  lastBetSize = 0,
  winners = [],
  stage = 0,
  highlightedCards = [],
  checkAction = '',
  onClearCheckAction,
  isGamePaused = false,
  onResumeGame,
  isTooltipVisible,
  setIsTooltipVisible,
  playersToAct
}) {
  // Responsive oval size: always fits window, no scroll
  const [ovalSize, setOvalSize] = useState({ width: 900, height: 540 });
  useEffect(() => {
    function handleResize() {
      // Use 95vw x 70vh for the oval, minus a small margin
      const marginW = window.innerWidth * 0.05;
      const marginH = window.innerHeight * 0.05;
      const w = Math.max(window.innerWidth - marginW, 320);
      const h = Math.max(window.innerHeight - marginH, 200);
      setOvalSize({ width: w, height: h });
    }
    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);
  
  // Arrange players evenly around the oval, main player always at bottom
  const arranged = getPlayerPositionsOval(players, mainPlayerIndex);

  // Find winner's positions for chip animation (can be multiple winners for split pot)
  const winnerPositions = winners.length > 0
    ? winners.map(w => arranged.find(({ player }) => player.name === w)).filter(Boolean)
    : [];
  const winnerAngles = winnerPositions.map(pos => pos.angle);

  // Handle click on poker table to resume game
  const handleTableClick = (e) => {
    // Don't resume if clicking on the question mark
    if (e.target.closest('.check-action-question-mark')) {
      return;
    }
    
    if (isGamePaused && onResumeGame) {
      onResumeGame();
    }
  };

  // Find the main player for ActionPanel
  const mainPlayer = players.find(player => player.is_main_player);

  // Render the poker table
  return (
    <div className={`poker-table ${isGamePaused ? 'game-paused' : ''}`} onClick={handleTableClick}>
      {/* Table felt oval background, always smaller than player oval */}
      <div className="table-felt" style={{ width: '60%', height: '60%' }}></div>
      {/* Pot and community cards in center */}
      <div className="table-center" style={{ left: '50%', top: '50%' }}>
        <div className="pot" style={{ fontSize: '0.8rem', borderRadius: '8px', marginBottom:'12rem', marginTop: '0.52rem'}}>Pot: {formatNumber(pot)}BB</div>
        {/* Show chips in the pot */}
        {pot > 0 && (
          <ChipStack 
            amount={pot} 
            position="center" 
            winners={winners}
            winnerAngles={winnerAngles}
            ovalSize={ovalSize}
          />
        )}
        <div className="community-cards" style={{ gap: '0.18rem', marginTop:'-10rem' }}>
          {communityCards.length > 0 && communityCards[0] != undefined && communityCards.filter(Boolean).map((card, i) => (
            <Card 
              key={i} 
              card={card} 
              hidden={false} 
              isGrayedOut={highlightedCards.length > 0 && !highlightedCards.some(hc => hc.rank === card.rank && hc.suit === card.suit)}
            />  
          ))}
        </div>
      </div>
      {/* Players around the oval */}
      <div className="players-oval" style={{ width: ovalSize.width, height: ovalSize.height }}>
        {arranged.map(({ player, angle }, idx) => (
          <div
            key={player.id}
            style={{
              position: 'absolute',
              ...getOvalPosition(angle, ovalSize.width, ovalSize.height),
              pointerEvents: 'auto',
              zIndex: 10,
              transform: 'translate(-50%, -50%)',
            }}
          >
            <Player
              player={player}
              isMain={player.is_main_player}
              isCardsHidden={winners.length > 0 ? false : !player.is_main_player}
              isPlayerTurn={playersToAct && playersToAct.length > 0 && playersToAct[0] === player.name}
              isNeedToAct={player.is_need_to_act}
              positionLabel={player.position}
              mainPlayerActionHandler={mainPlayerActionHandler}
              isMainPlayerActive={isMainPlayerActive}
              lastBetSize={lastBetSize}
              playerAngle={angle}
              stage={stage}
              pot={pot}
              highlightedCards={highlightedCards}
              checkAction={checkAction}
              onClearCheckAction={onClearCheckAction}
              isGamePaused={isGamePaused}
              onResumeGame={onResumeGame}
              isTooltipVisible={isTooltipVisible}
              setIsTooltipVisible={setIsTooltipVisible}
            />
          </div>
        ))}
      </div>
      
      {/* Action Panel for main player */}
      {mainPlayer && !mainPlayer.is_folded && (
        <ActionPanel
          isMainPlayerActive={isMainPlayerActive}
          lastBetSize={lastBetSize}
          playerStack={mainPlayer.stack_size}
          playerChipsInPot={mainPlayer.chips_in_pot}
          mainPlayerActionHandler={mainPlayerActionHandler}
          stage={stage}
          pot={pot}
        />
      )}
    </div>
  );
}

export default PokerTable; 
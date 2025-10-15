import React from 'react';
import './css_files/Card.css';

function Card({ card, hidden = false, isGrayedOut = false }) {  
  // Handle cases where card is undefined, null, or invalid
  if (!card || typeof card !== 'object') {
    console.log(card)
    console.log('Card is invalid, showing card back');
    return (
      <div className="card card-back">
        ♠
      </div>
    );
  }

  if (hidden) {
    return (
      <div className="card card-back">
        <div className="card-back-symbol">♠</div>
      </div>
    );
  }

  // Safely destructure card properties with fallbacks
  const { rank, suit } = card;
  
  // Check if rank and suit are valid
  if (!rank || !suit) {
    console.log('Card missing rank or suit:', { rank, suit });
    return (
      <div className="card card-back">
        ♠
      </div>
    );
  }

  const isRed = suit === '♥' || suit === '♦';

  return (
    <div className={`card ${isRed ? 'red' : ''} ${isGrayedOut ? 'highlighted-card' : ''}`}>
      <div className="card-content">
        <div className="card-rank">{rank}</div>
        <div className="card-suit">{suit}</div>
      </div>
    </div>
  );
}

export default Card; 
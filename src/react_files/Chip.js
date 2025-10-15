import React from 'react';
import './css_files/Chip.css';

const Chip = ({ value, color = 'red', isAnimating = false, animationDelay = 0, winnerPos = null }) => {
  const chipStyle = isAnimating && winnerPos ? {
    animationDelay: `${animationDelay}s`,
    '--winner-x': `${winnerPos.x}px`,
    '--winner-y': `${winnerPos.y}px`,
  } : {};

  return (
    <div 
      className={`chip chip-${color} ${isAnimating ? 'chip-moving-to-winner' : ''}`}
      style={chipStyle}
    >
      <div className="chip-value">{value}</div>
    </div>
  );
};

export default Chip; 
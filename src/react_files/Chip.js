import React from 'react';
import './css_files/Chip.css';

const Chip = ({ value, color = 'red', isAnimating = false, animationDelay = 0, winnerPos = null }) => {
  const colorMap = {
    red: ['#ff6666', '#cc0000', '#990000'],
    blue: ['#6666ff', '#0000cc', '#000099'],
    green: ['#66ff66', '#00cc00', '#009900'],
    black: ['#666666', '#333333', '#000000'],
    purple: ['#aa66ff', '#8844ff', '#4400cc'],
    yellow: ['#ffdd44', '#ffcc00', '#ddaa00'],
    white: ['#ffffff', '#e0e0e0', '#cccccc'],
    silver: ['#e8e8e8', '#c0c0c0', '#a0a0a0']
  };
  
  const colors = colorMap[color] || colorMap.red;
  const chipStyle = {
    '--chip-color-1': colors[0],
    '--chip-color-2': colors[1],
    '--chip-color-3': colors[2],
    ...(isAnimating && winnerPos ? {
      animationDelay: `${animationDelay}s`,
      '--winner-x': `${winnerPos.x}px`,
      '--winner-y': `${winnerPos.y}px`,
    } : {})
  };

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
import React from 'react';
import Chip from './Chip';
import './css_files/ChipStack.css';

// Helper function to format numbers - show decimals only if they exist
const formatNumber = (value) => {
  const num = parseFloat(value);
  if (isNaN(num)) return '0';
  // Round to 2 decimal places to handle floating point precision issues
  const rounded = Math.round(num * 100) / 100;
  return rounded % 1 === 0 ? rounded.toString() : rounded.toFixed(2);
};

const ChipStack = ({ amount, position = 'center', winners = [], winnerAngles = [], ovalSize = null }) => {
  // Convert amount to chips (standard poker chip values)
  const getChips = (amount) => {
    const chipStacks = [];
    let remaining = amount;
    
    // Standard chip values (you can adjust these)
    const chipValues = [25, 10, 5, 1];
    const chipColors = ['black', 'blue', 'red', 'white'];
    
    for (let i = 0; i < chipValues.length && remaining > 0; i++) {
      const value = chipValues[i];
      const count = Math.floor(remaining / value);
      if (count > 0) {
        chipStacks.push({ value, color: chipColors[i], count });
        remaining -= count * value;
      }
    }
    
    // If there's still remaining, add it as a single chip
    if (remaining > 0) {
      chipStacks.push({ value: remaining, color: 'green', count: 1 });
    }
    
    return chipStacks;
  };

  const chipStacks = getChips(parseFloat(amount) || 0);

  if (chipStacks.length === 0) return null;

  // Calculate winner positions for animation (can be multiple for split pot)
  const getWinnerPositions = () => {
    if (winners.length === 0 || !winnerAngles || winnerAngles.length === 0 || !ovalSize) return [];
    
    const cx = ovalSize.width / 2;
    const cy = ovalSize.height / 2;
    const rx = ovalSize.width * 0.4;
    const ry = ovalSize.height * 0.37;
    
    return winnerAngles.map(angle => ({
      x: cx + rx * Math.cos(angle) - cx,
      y: cy - ry * Math.sin(angle) - cy
    }));
  };

  const winnerPositions = getWinnerPositions();
  const isAnimating = winners.length > 0 && winnerPositions.length > 0;

  return (
    <div className={`chip-stack-container chip-position-${position} ${isAnimating ? 'chip-animating-to-winner' : ''}`}>
      <div className="chip-stack">
        {chipStacks.map((stack, index) => (
          <div key={index} className={stack.count > 1 ? "chip-stack-group" : "chip-single"}>
            {Array.from({ length: stack.count }, (_, i) => {
              // For split pot, distribute chips to different winners
              const winnerIndex = winnerPositions.length > 0 ? i % winnerPositions.length : 0;
              const targetPos = winnerPositions[winnerIndex];
              
              return (
                <Chip 
                  key={i} 
                  value={formatNumber(stack.value)} 
                  color={stack.color}
                  isAnimating={isAnimating}
                  animationDelay={index * 0.1 + i * 0.05}
                  winnerPos={targetPos}
                />
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
};

export default ChipStack; 
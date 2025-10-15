import React, { useState, useEffect, useRef } from 'react';
import './css_files/ActionPanel.css';

// ActionPanel component for player actions (raise/call/fold)
function ActionPanel({ 
  isMainPlayerActive, 
  lastBetSize, 
  playerStack, 
  playerChipsInPot, 
  mainPlayerActionHandler, 
  stage = 0, 
  pot = 0 
}) {
  // State for raise amount
  const [raiseAmount, setRaiseAmount] = useState(0);
  const [sliderValue, setSliderValue] = useState(50); // 0-100 for slider
  // State for auto-action
  const [autoAction, setAutoAction] = useState(false);
  const autoActionRef = useRef(false);

  // Calculate min and max raise amounts
  const minRaise = lastBetSize + 1;
  const maxRaise = playerStack + playerChipsInPot;
  const minRaiseSlider = minRaise;
  const maxRaiseSlider = maxRaise;

  useEffect(() => {
    autoActionRef.current = autoAction;
  }, [autoAction]);

  // Update raise amount when slider changes
  useEffect(() => {
    const calculatedValue = stage === 0 
      ? Math.min((lastBetSize * 2.5), playerStack) 
      : Math.min(playerStack, (pot * 0.5));
    
    // Set initial slider value based on calculated value
    const initialSliderValue = Math.max(0, Math.min(100, 
      ((calculatedValue - minRaiseSlider) / (maxRaiseSlider - minRaiseSlider)) * 100
    ));
    setSliderValue(initialSliderValue);
    setRaiseAmount(calculatedValue);
  }, [lastBetSize, stage, pot, playerStack, minRaiseSlider, maxRaiseSlider]);

  useEffect(() => {
    setAutoAction(false);
  }, [stage, playerChipsInPot]);

  // Handle slider change
  const handleSliderChange = (e) => {
    const sliderPercent = parseInt(e.target.value);
    setSliderValue(sliderPercent);
    
    // Calculate raise amount based on slider percentage
    const calculatedRaise = minRaiseSlider + (sliderPercent / 100) * (maxRaiseSlider - minRaiseSlider);
    setRaiseAmount(Math.round(calculatedRaise));
  };

  // Handle raise button click
  const handleRaise = () => {
    if (raiseAmount >= minRaise && raiseAmount <= maxRaise) {
      mainPlayerActionHandler('RAISE', raiseAmount);
    }
  };

  // Handle call button click
  const handleCall = () => {
    mainPlayerActionHandler('CALL', lastBetSize);
  };

  // Handle fold/check button click
  const handleFoldOrCheck = () => {
    if (lastBetSize <= playerChipsInPot) {
      mainPlayerActionHandler('CHECK', 0);
    } else {
      mainPlayerActionHandler('FOLD', 0);
    }
  };

  // Auto-action logic - execute when auto-action is enabled and it's the player's turn
  useEffect(() => {
    if (autoAction && isMainPlayerActive) {
      // Small delay to allow UI to update
      const timer = setTimeout(() => {
        if (lastBetSize <= playerChipsInPot) {
          mainPlayerActionHandler('CHECK', 0);
        } else {
          mainPlayerActionHandler('FOLD', 0);
        }
      }, 500);
      
      return () => clearTimeout(timer);
    }
  }, [autoAction, isMainPlayerActive, lastBetSize, playerChipsInPot, mainPlayerActionHandler]);

  // Format number for display
  const formatNumber = (value) => {
    const num = parseFloat(value);
    if (isNaN(num)) return '0';
    const rounded = Math.round(num * 100) / 100;
    return rounded % 1 === 0 ? rounded.toString() : rounded.toFixed(2);
  };

  return (
    <div className="action-panel">
      {/* Action Buttons - only show when player is active */}
      {isMainPlayerActive && !autoActionRef.current && (
        <div className="action-buttons">
          <button 
            className="action-btn fold-check-btn" 
            onClick={handleFoldOrCheck} 
          >
            {lastBetSize <= playerChipsInPot ? 'Check' : 'Fold'}
          </button>
          
          {lastBetSize > 0 && lastBetSize > playerChipsInPot && (
            <button 
              className="action-btn call-btn" 
              onClick={handleCall} 
            >
              Call {formatNumber(lastBetSize-playerChipsInPot)}BB
            </button>
          )}
          
          <button 
            className="action-btn raise-btn" 
            onClick={handleRaise} 
            disabled={raiseAmount < minRaise}
          >
            Raise {formatNumber(raiseAmount)}BB
          </button>
        </div>
      )}

      {/* Vertical Raise Slider - only show when player is active */}
      {isMainPlayerActive && !autoActionRef.current && (
        <div className="raise-slider-container">
          <div className="raise-amount-display">
            {formatNumber(raiseAmount)}BB
          </div>
          <input
            type="range"
            min="0"
            max="100"
            value={sliderValue}
            onChange={handleSliderChange}
            className="raise-slider vertical"
            orient="vertical"
          />
          <div className="slider-labels">
            <span>{formatNumber(minRaise)}BB</span>
          </div>
        </div>
      )}

      {/* Auto-action Checkbox - only show when player is NOT active */}
      {!isMainPlayerActive && (
        <div className="auto-action-checkbox">
          <label>
            <input
              type="checkbox"
              checked={autoAction}
              onChange={(e) => setAutoAction(e.target.checked)}
            />
            <span>
              {lastBetSize <= playerChipsInPot ? 'Check/Fold' : 'Fold'}
            </span>
          </label>
        </div>
      )}
    </div>
  );
}

export default ActionPanel; 
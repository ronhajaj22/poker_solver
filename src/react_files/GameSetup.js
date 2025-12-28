import React, { useState, useEffect, useRef } from 'react';
import './css_files/GameSetup.css';

// GameSetup component renders the initial setup screen for the poker app
// Allows the user to select game options and start the game
function GameSetup({ onStartGame }) {
  // State for each setup option
  const [numPlayers, setNumPlayers] = useState(2); // Default: 6 players
  const [stackSize, setStackSize] = useState(100); // Default: 100 chips
  const [clubMode, setClubMode] = useState(false); // Default: Exploiver Solver
  const [showSettings, setShowSettings] = useState(false); // Settings menu visibility
  const [tempSettings, setTempSettings] = useState({
    soundEnabled: true,
    foldToEndGame: false,
    animationsEnabled: true,
    autoSave: true,
    theme: 'dark',
    animationSpeed: 'normal',
    defaultBlindSize: 2,
    defaultNumPlayers: 2,
    defaultStackSize: 100,
    foldToEndGame: false,
    experienceLevel: 'Just Starting'
  });
  
  const settingsRef = useRef(null);
  const buttonRef = useRef(null);

  // Initialize form values from settings
  useEffect(() => {
    setNumPlayers(tempSettings.defaultNumPlayers);
    setStackSize(tempSettings.defaultStackSize);
  }, []);

  // Handle clicks outside the settings menu
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (showSettings && 
          settingsRef.current && 
          !settingsRef.current.contains(event.target) &&
          buttonRef.current && 
          !buttonRef.current.contains(event.target)) {
        setShowSettings(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [showSettings]);

  // Handle form submission
  const handleSubmit = (e) => {
    e.preventDefault(); // Prevent page reload
    // Call the parent handler to start the game with selected options
    onStartGame(numPlayers, stackSize, clubMode);
  };

  // Handle settings changes
  const handleSettingChange = (setting, value) => {
    setTempSettings(prev => ({
      ...prev,
      [setting]: value
    }));
  };

  // Apply settings and close menu
  const handleConfirmSettings = () => {
    // Update main form values based on settings
    setNumPlayers(tempSettings.defaultNumPlayers);
    setStackSize(tempSettings.defaultStackSize);
    setShowSettings(false);
  };

  // Cancel settings changes
  const handleCancelSettings = () => {
    // Reset temp settings to current values
    setTempSettings(prev => ({
      ...prev,
      defaultBlindSize: 1,
      defaultNumPlayers: numPlayers,
      defaultStackSize: stackSize
    }));
    setShowSettings(false);
  };

  // Toggle settings menu
  const toggleSettings = () => {
    setShowSettings(!showSettings);
  };

  // Render the setup form
  return (
    <div className="game-setup">
      {/* Settings Button */}
      <div className="settings-container">
        <button 
          ref={buttonRef}
          className="settings-button" 
          onClick={toggleSettings}
          aria-label="Settings"
        >
          ⚙️
        </button>
        
        {/* Settings Menu */}
        {showSettings && (
          <div ref={settingsRef} className="settings-menu">
            <h3>Settings</h3>
            
            <div className="setting-item">
              <span>Sound Effects</span>
              <input
                type="checkbox"
                checked={tempSettings.soundEnabled}
                onChange={(e) => handleSettingChange('soundEnabled', e.target.checked)}
              />
            </div>
            
            <div className="setting-item">
              <span>Fold to End Game</span>
              <input
                type="checkbox"
                checked={tempSettings.foldToEndGame}
                onChange={(e) => handleSettingChange('foldToEndGame', e.target.checked)}
              />
            </div>
            
            <div className="setting-item">
              <span>Animation Speed:</span>
              <select
                value={tempSettings.animationSpeed}
                onChange={(e) => handleSettingChange('animationSpeed', e.target.value)}
              >
                <option value="slow">Slow</option>
                <option value="normal">Normal</option>
                <option value="fast">Fast</option>
              </select>
            </div>
            
            <div className="setting-item">
              <span>Default Players:</span>
              <select
                value={tempSettings.defaultNumPlayers}
                onChange={(e) => handleSettingChange('defaultNumPlayers', Number(e.target.value))}
              >
                {[2,3,4,5,6,7,8,9].map(n => (
                  <option key={n} value={n}>{n}</option>
                ))}
              </select>
            </div>
            
            <div className="setting-item">
              <span>Default Stack Size:</span>
              <select
                value={tempSettings.defaultStackSize}
                onChange={(e) => handleSettingChange('defaultStackSize', Number(e.target.value))}
              >
                <option value={20}>20</option>
                <option value={50}>50</option>
                <option value={100}>100</option>
                <option value={200}>200</option>
                <option value={500}>500</option>
              </select>
            </div>
            
            <div className="setting-item">
              <span>Experience Level:</span>
              <select
                value={tempSettings.experienceLevel}
                onChange={(e) => handleSettingChange('experienceLevel', e.target.value)}
              >
                <option value="Just Starting">Just Starting</option>
                <option value="Mid stakes">Mid stakes</option>
                <option value="Pro">Pro</option>
                <option value="Mathematician">Mathematician</option>
              </select>
            </div>
            
            <div className="setting-item">
              <span>Cards Style:</span>
              <select
                value={tempSettings.card_style}
                onChange={(e) => handleSettingChange('card_style', e.target.value)}
              >
                <option value="normal">Normal</option>
                <option value="color">Color</option>
              </select>
            </div>

            <div className="settings-buttons">
              <button className="settings-btn cancel-btn" onClick={handleCancelSettings}>
                Cancel
              </button>
              <button className="settings-btn confirm-btn" onClick={handleConfirmSettings}>
                Confirm
              </button>
            </div>
          </div>
        )}
      </div>

      <h2>Start New Game</h2>
      <form onSubmit={handleSubmit}>
        {/* Number of Players dropdown */}
        <label>
          Number of Players:
          <select value={numPlayers} onChange={e => setNumPlayers(Number(e.target.value))}>
            {[2,3,4,5,6,7,8,9].map(n => (
              <option key={n} value={n}>{n}</option>
            ))}
          </select>
        </label>
        {/* Stack Size dropdown */}
        <label>
          Stack Size:
          <select value={stackSize} onChange={e => setStackSize(Number(e.target.value))}>
            {[20, 50, 100, 200, 500].map(n => (
              <option key={n} value={n}>{n}</option>
            ))}
          </select>
        </label>
        {/* Opponent Type dropdown */}
        <label>
          <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            Opponent Type:
            <span 
              className={clubMode ? "info-icon info-icon-club" : "info-icon info-icon-exploitive"}               
              title={clubMode ? "In this mode, the solver plays like the average player in your club and shows how to improve in order to exploit your opponents as effectively as possible." : "In this mode, the solver learns your tendencies and weaknesses, then shows where you make mistakes and how to improve your game."}
            >
              ?
            </span>
          </span>
          <select value={clubMode ? 'club' : 'exploitive'} onChange={e => setClubMode(e.target.value === 'club')}>
            <option value="exploitive">Exploitive Solver</option>
            <option value="club">Club Player</option>
          </select>
        </label>
        
        {/* Start Game button */}
        <button type="submit" className={clubMode ? "button button-club" : "button button-exploitive"}>Start Game</button>
      </form>
    </div>
  );
}

export default GameSetup; 
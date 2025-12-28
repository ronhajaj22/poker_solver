// Sound utility functions for poker game

// Play a card flip sound (for flop/turn/river)
export const playFlopSound = (plays = 1) => {
  try {
    // Pre-load all audio objects
    const audios = [];
    for (let i = 0; i < plays; i++) {
      const audio = new Audio('/sounds/flipcard.mp3');
      audio.volume = 0.5;
      audio.playbackRate = 1.1; // Speed up playback (1 = normal, 2 = double speed)
      audios.push(audio);
    }
    
    // Play each with a delay using setTimeout scheduling
    audios.forEach((audio, index) => {
      setTimeout(() => {
        audio.play();
      }, index * 450); // 500ms delay between each card
    });
  } catch (error) {
    console.warn('Could not play flop sound:', error);
  }
};

export const playBetSound = () => {
    try {
        const audio = new Audio('/sounds/betsound.mp3');
        audio.volume = 0.5;
        audio.play();
      } catch (error) {
        console.warn('Could not play bet sound:', error);
      }
};

// Play a check sound (tap on table)
export const playCheckSound = () => {
  try {
    const audio = new Audio('/sounds/fast-door-knocking.mp3');
    audio.volume = 0.5;
    audio.play().then(() => {
      // Wait for play to start, then pause after 100ms
      setTimeout(() => {
        audio.pause();
      }, 400);
    }).catch(err => console.warn('Audio play failed:', err));
  } catch (error) {
    console.warn('Could not play check sound:', error);
  }
};
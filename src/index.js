import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './react_files/App';
import './react_files/css_files/index.css';

// Render the App component into the root div
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
); 
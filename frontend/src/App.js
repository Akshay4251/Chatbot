import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [input, setInput] = useState('');
  const [chatLog, setChatLog] = useState([]);

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage = { sender: 'user', text: input };
    setChatLog([...chatLog, userMessage]);

    try {
      const response = await axios.post('http://localhost:5000/api/chat', {
        message: input
      });

      const botMessage = { sender: 'bot', text: response.data.response };
      setChatLog([...chatLog, userMessage, botMessage]);
    } catch (err) {
      setChatLog([...chatLog, { sender: 'bot', text: "Error contacting bot." }]);
    }

    setInput('');
  };

  return (
    <div className="chat-container">
      <div className="chat-log">
        {chatLog.map((msg, i) => (
          <div key={i} className={`message ${msg.sender}`}>
            <strong>{msg.sender === 'user' ? 'You' : 'Bot'}:</strong> {msg.text}
          </div>
        ))}
      </div>
      <div className="input-area">
        <input
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && handleSend()}
          placeholder="Type a message..."
        />
        <button onClick={handleSend}>Send</button>
      </div>
    </div>
  );
}

export default App;

import React, { useState, useEffect } from 'react';

const TypingEffect = ({ text, speed = 30 }) => {
  const [displayedText, setDisplayedText] = useState('');
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isComplete, setIsComplete] = useState(false);

  useEffect(() => {
    // Reset if text changes
    setDisplayedText('');
    setCurrentIndex(0);
    setIsComplete(false);
  }, [text]);

  useEffect(() => {
    if (currentIndex < text.length) {
      const timeout = setTimeout(() => {
        setDisplayedText(prev => prev + text[currentIndex]);
        setCurrentIndex(prevIndex => prevIndex + 1);
      }, speed);

      return () => clearTimeout(timeout);
    } else if (!isComplete) {
      setIsComplete(true);
    }
  }, [currentIndex, text, speed, isComplete]);

  // Split the text into lines and map each line to a paragraph
  const renderText = () => {
    return displayedText.split('\n').map((line, i) => (
      <p key={i}>{line || <br/>}</p>
    ));
  };

  return (
    <div className="typing-effect">
      {renderText()}
      {!isComplete && <span className="typing-cursor">|</span>}
    </div>
  );
};

export default TypingEffect;

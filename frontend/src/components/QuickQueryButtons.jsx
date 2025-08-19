import React from 'react';

const QuickQueryButtons = ({ onQuerySelect }) => {
  const queries = [
    {
      category: "Weather",
      questions: [
        "Will it rain this week?",
        "What's the soil moisture level?",
        "Should I irrigate my field today?",
      ]
    },
    {
      category: "Crops",
      questions: [
        "Which crop should I plant this season?",
        "Best varieties for drought resistance?",
        "How to improve my crop yield?",
      ]
    },
    {
      category: "Market",
      questions: [
        "Current market price for wheat?",
        "When should I sell my harvest?",
        "How are soybean prices trending?",
      ]
    },
    {
      category: "Pests & Disease",
      questions: [
        "How to control aphids naturally?",
        "My crop has yellow leaves, what could it be?",
        "Best practices for pest management?",
      ]
    }
  ];

  return (
    <div className="quick-query-container">
      {queries.map((category, idx) => (
        <div key={idx} className="query-category">
          <h4>{category.category}</h4>
          <div className="query-buttons">
            {category.questions.map((question, i) => (
              <button 
                key={i} 
                onClick={() => onQuerySelect(question)}
                className="query-button"
              >
                {question}
              </button>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
};

export default QuickQueryButtons;

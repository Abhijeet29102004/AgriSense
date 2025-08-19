import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './components.css';

const WeatherWidget = ({ lat, lon, locationStatus, onClose }) => {
  const [weather, setWeather] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [address, setAddress] = useState('');

  useEffect(() => {
    const fetchWeather = async () => {
      if (!lat || !lon) return;
      
      setLoading(true);
      try {
        // We'll call our backend that wraps the weather API
        const response = await axios.get('http://127.0.0.1:8000/weather', {
          params: { lat, lon }
        });
        
        if (response.data) {
          setWeather(response.data);
          
          // Try to get address from coordinates
          try {
            const geoResponse = await axios.get('http://127.0.0.1:8000/geocode', {
              params: { lat, lon }
            });
            
            if (geoResponse.data && geoResponse.data.address) {
              setAddress(geoResponse.data.address);
            }
          } catch (geoErr) {
            console.error('Failed to get location name:', geoErr);
          }
        }
      } catch (err) {
        console.error('Failed to fetch weather:', err);
        setError('Failed to load weather data. Please try again later.');
      } finally {
        setLoading(false);
      }
    };

    if (locationStatus === 'success') {
      fetchWeather();
    }
  }, [lat, lon, locationStatus]);

  // Format temperature for display
  const formatTemp = (temp) => {
    return temp ? `${Math.round(temp)}°C` : '--';
  };

  // Format date as "Day, Month Date"
  const formatDate = (timestamp) => {
    if (!timestamp) return '';
    const date = new Date(timestamp * 1000);
    return date.toLocaleDateString('en-US', { 
      weekday: 'short', 
      month: 'short', 
      day: 'numeric' 
    });
  };

  // Get weather icon URL
  const getIconUrl = (icon) => {
    return icon ? `https://www.weatherbit.io/static/img/icons/${icon}.png` : '';
  };

  if (locationStatus === 'pending' || locationStatus === 'loading') {
    return (
      <div className="weather-widget">
        <div className="weather-header">
          <h3>Weather Forecast</h3>
          <button className="close-btn" onClick={onClose}>&times;</button>
        </div>
        <div className="weather-content loading">
          <div className="loading-spinner"></div>
          <p>Loading weather data...</p>
        </div>
      </div>
    );
  }

  if (locationStatus === 'error') {
    return (
      <div className="weather-widget">
        <div className="weather-header">
          <h3>Weather Forecast</h3>
          <button className="close-btn" onClick={onClose}>&times;</button>
        </div>
        <div className="weather-content error">
          <p>Location access is needed for accurate weather data.</p>
          <button 
            className="retry-btn"
            onClick={() => window.location.reload()}
          >
            Allow Location Access
          </button>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="weather-widget">
        <div className="weather-header">
          <h3>Weather Forecast</h3>
          <button className="close-btn" onClick={onClose}>&times;</button>
        </div>
        <div className="weather-content error">
          <p>{error}</p>
        </div>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="weather-widget">
        <div className="weather-header">
          <h3>Weather Forecast</h3>
          <button className="close-btn" onClick={onClose}>&times;</button>
        </div>
        <div className="weather-content loading">
          <div className="loading-spinner"></div>
          <p>Loading weather data...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="weather-widget">
      <div className="weather-header">
        <h3>Weather Forecast</h3>
        <button className="close-btn" onClick={onClose}>&times;</button>
      </div>

      {weather ? (
        <div className="weather-content">
          <div className="weather-location">
            <h4>{address || 'Your Location'}</h4>
          </div>

          {/* Current weather */}
          {weather.current && (
            <div className="current-weather">
              <div className="temp-icon">
                <img 
                  src={getIconUrl(weather.current.weather?.icon)} 
                  alt={weather.current.weather?.description || 'Weather'} 
                  className="weather-icon"
                />
                <span className="temp">{formatTemp(weather.current.temp)}</span>
              </div>
              <div className="weather-details">
                <p className="weather-desc">{weather.current.weather?.description}</p>
                <div className="weather-metrics">
                  <div>
                    <span>Humidity: </span>
                    <span>{weather.current.rh}%</span>
                  </div>
                  <div>
                    <span>Wind: </span>
                    <span>{weather.current.wind_spd?.toFixed(1)} m/s</span>
                  </div>
                  {weather.current.precip !== undefined && (
                    <div>
                      <span>Precip: </span>
                      <span>{weather.current.precip} mm</span>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Forecast */}
          {weather.forecast && weather.forecast.length > 0 && (
            <div className="forecast">
              <h4>7-Day Forecast</h4>
              <div className="forecast-days">
                {weather.forecast.slice(0, 7).map((day, index) => (
                  <div key={index} className="forecast-day">
                    <div className="day-name">{formatDate(day.ts)}</div>
                    <img 
                      src={getIconUrl(day.weather?.icon)} 
                      alt={day.weather?.description} 
                      className="forecast-icon"
                    />
                    <div className="forecast-temp">
                      <span className="max">{formatTemp(day.max_temp)}</span>
                      <span className="min">{formatTemp(day.min_temp)}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="farming-tips">
            <h4>Farming Recommendations</h4>
            <ul>
              {weather.current?.precip > 5 ? (
                <li>Heavy rain expected. Consider postponing outdoor activities.</li>
              ) : weather.current?.precip > 0 ? (
                <li>Light rain expected. Good time for planting.</li>
              ) : (
                <li>No rain expected. Ensure proper irrigation.</li>
              )}
              
              {weather.current?.temp > 35 ? (
                <li>High temperature. Ensure crops have shade and extra water.</li>
              ) : weather.current?.temp < 10 ? (
                <li>Low temperature. Protect sensitive crops from frost.</li>
              ) : (
                <li>Temperature is optimal for most crops.</li>
              )}
              
              {weather.current?.rh > 80 ? (
                <li>High humidity. Watch for fungal diseases.</li>
              ) : weather.current?.rh < 30 ? (
                <li>Low humidity. Increase watering frequency.</li>
              ) : (
                <li>Humidity levels are good for plant growth.</li>
              )}
            </ul>
          </div>

          <div className="weather-footer">
            <small>Data updated at: {new Date().toLocaleTimeString()}</small>
            <button 
              className="refresh-btn" 
              onClick={() => window.location.reload()}
              title="Refresh weather data"
            >
              ↻
            </button>
          </div>
        </div>
      ) : (
        <div className="weather-content error">
          <p>No weather data available.</p>
        </div>
      )}
    </div>
  );
};

export default WeatherWidget;

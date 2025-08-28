import React, { useState, useEffect, useRef } from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import axios from "axios";
import { v4 as uuidv4 } from "uuid";
import "./App.css";
import farmer from "./assets/farmer.jpg";
import { FaMapMarkerAlt } from "react-icons/fa";
import { MapContainer, TileLayer, Marker, useMapEvents } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';
import { Login, Signup, OAuthCallback } from "./pages/login-signup.jsx";
import Sidebar from "./components/sidebar.jsx";
import ChatBox from "./components/Chat.jsx";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";




delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-shadow.png',
});

const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

function LocationMarker({ position, setPosition, mapRef }) {
  const map = useMapEvents({
    click(e) {
      setPosition([e.latlng.lat, e.latlng.lng]);
      map.setView(e.latlng, 12); // Changed from 20 to 12 for a more reasonable zoom level
    },
  });



  // Store the map instance in the ref
  useEffect(() => {
    if (map && mapRef) {
      mapRef.current = map;
    }
  }, [map, mapRef]);

  return position ? <Marker position={position} /> : null;
}

function getUserFromToken() {
  const token = localStorage.getItem("token");
  if (!token) return null;
  try {
    const payload = JSON.parse(atob(token.split(".")[1]));
    return {
      user_id: payload.user_id || payload.id || payload.sub || payload.gmail || null,
      name: payload.name || (payload.gmail ? payload.gmail.split("@")[0] : "User"),
      gmail: payload.gmail || null,
    };
  } catch {
    return null;
  }
}

function MainApp() {

  
  const [userId] = useState(() => uuidv4());
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState([]);
  const [darkMode, setDarkMode] = useState(false);
  const [location, setLocation] = useState({ lat: null, lon: null });
  const [listening, setListening] = useState(false);
  const [lang, setLang] = useState("en");
  const [isLoading, setIsLoading] = useState(false);
  const [showWelcome, setShowWelcome] = useState(false);
  const [lastUserQuery, setLastUserQuery] = useState("");
  const [showLocationModal, setShowLocationModal] = useState(false);
  const [mapPosition, setMapPosition] = useState([20.5937, 78.9629]); // Default to center of India
  const [locationName, setLocationName] = useState("");
  const [locationSearch, setLocationSearch] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [user, setUser] = useState(getUserFromToken());
  const [chats, setChats] = useState([]);
  const [activeChatId, setActiveChatId] = useState(null);

  const chatWindowRef = useRef(null);
  const mapRef = useRef(null);
  const chatBoxRef = useRef(null);
  const locationToastShown = useRef(false); // <-- Add this line
  const prevUser = useRef(null);

  const formatLocation = () => {
    if (!location.lat || !location.lon) return "Location not set";
    return `${location.lat.toFixed(3)}°, ${location.lon.toFixed(3)}°`;
  };

  const toggleSidebar = () => setSidebarOpen(prev => !prev);

  // Handle location selection from map
  const handleLocationSearch = async (e) => {
  e.preventDefault();
  if (!locationSearch.trim()) return;
  
  try {
    const response = await axios.get(
      `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(locationSearch)}&limit=1`
    );
    
    if (response.data && response.data.length > 0) {
      const { lat, lon } = response.data[0];
      const newPosition = [parseFloat(lat), parseFloat(lon)];
      setMapPosition(newPosition);
      
      // Set appropriate zoom level based on the result type
      const zoomLevel = determineZoomLevel(response.data[0].type);
      
      // Use the map reference to update view with proper zoom
      if (mapRef.current) {
        mapRef.current.setView(newPosition, zoomLevel);
      }
    } else {
      alert("Location not found. Please try a different search term.");
    }
  } catch (error) {
    console.error("Error searching for location:", error);
    alert("Error searching for location. Please try again.");
  }
};

  // --- LOGOUT HANDLER ---
  const handleLogout = () => {
    localStorage.removeItem("token");
    setUser(null);
    setChats([]);
    setActiveChatId(null);
    // Optionally clear other user-related state here
  };

const determineZoomLevel = (locationType) => {
  // Default zoom levels based on location type
  switch(locationType) {
    case 'country':
      return 6;
    case 'state':
    case 'administrative':
      return 8;
    case 'county':
    case 'district':
      return 15;
    case 'city':
    case 'town':
      return 14;
    case 'village':
    case 'suburb':
      return 15;
    case 'neighbourhood':
    case 'building':
      return 16;
    default:
      return 16;
  }
};

  // Get initial user location
  useEffect(() => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (pos) => {
          const newLocation = { lat: pos.coords.latitude, lon: pos.coords.longitude };
          setLocation(newLocation);
          setMapPosition([pos.coords.latitude, pos.coords.longitude]);
          fetchLocationName(newLocation.lat, newLocation.lon);
        },
        (err) => {
          if (!locationToastShown.current) {
            toast.info("Location is not set. Location will be used for better responses.", {
              position: "top-right",
              autoClose: 5000,
              hideProgressBar: false,
              closeOnClick: true,
              pauseOnHover: true,
              draggable: true,
              theme: "colored",
            });
            locationToastShown.current = true; // <-- Show only once
          }
          console.error("Location permission denied.", err);
        }
      );
    } else {
      console.warn("Geolocation not supported");
    }
  }, []);

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const res = await axios.get(`${backendUrl}/history`, {
          params: { user_id: userId, limit: 50 }
        });
        if (res.data.history) {
          const formatted = res.data.history.map(msg => ({
            sender: msg.role === "user" ? "user" : "ai",
            text: msg.content
          }));
          setMessages(formatted);
        }
      } catch (err) {
        console.error("Failed to fetch history:", err);
      }
    };
    fetchHistory();
  }, [userId]);

  // Refresh chats
  const refreshChats = async () => {
    if (user && user.user_id) {
      const res = await axios.get(`${backendUrl}/user/chats`, { params: { user_id: user.user_id } });
      setChats(res.data.chats);
      // Remove setActiveChatId here!
    }
  };

  useEffect(() => {
    refreshChats();
  }, [user]);

  // Function to get location name from coordinates
  const fetchLocationName = async (lat, lon) => {
    try {
      const response = await axios.get(
        `https://nominatim.openstreetmap.org/reverse?format=json&lat=${lat}&lon=${lon}&zoom=10`
      );
      const address = response.data.address;
      const locationDisplay = address.village || address.town || address.city || address.county || address.state || 'Current location';
      setLocationName(locationDisplay);
    } catch (error) {
      console.error("Error getting location name:", error);
      setLocationName("Current location");
    }
  };

  const scrollToBottom = () => {
    // Scroll chat window
    if (chatWindowRef.current) {
      chatWindowRef.current.scrollTop = chatWindowRef.current.scrollHeight;
    }
    
    // Scroll the entire page to the bottom of the chat container
    const chatContainer = document.querySelector('.chat-container');
    if (chatContainer) {
      chatContainer.scrollIntoView({ behavior: 'smooth', block: 'end' });
    }
  };

  // Auto-scroll when messages change
  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);
  
  const welcomeContent = {
  en: {
    title: "Welcome to KishanMitra!",
    message: "I'm your AI assistant for all farming questions. Ask me about crops, weather, pests, or agricultural practices. You can use voice input or quick questions to get started!",
    button: "Get Started"
  },
  hi: {
    title: "किशनमित्र में आपका स्वागत है!",
    message: "मैं खेती से संबंधित सभी प्रश्नों के लिए आपका AI सहायक हूं। फसलों, मौसम, कीटों या कृषि पद्धतियों के बारे में मुझसे पूछें। आप शुरू करने के लिए आवाज इनपुट या त्वरित प्रश्नों का उपयोग कर सकते हैं!",
    button: "शुरू करें"
  },
  bn: {
    title: "কিষানমিত্র-এ আপনাকে স্বাগতম!",
    message: "আমি কৃষি সম্পর্কিত সব প্রশ্নের জন্য আপনার এআই সহকারী। ফসল, আবহাওয়া, পোকামাকড় বা কৃষি পদ্ধতি সম্পর্কে আমাকে জিজ্ঞাসা করুন। শুরু করতে ভয়েস ইনপুট বা দ্রুত প্রশ্ন ব্যবহার করুন!",
    button: "শুরু করুন"
  },
  te: {
    title: "కిసాన్ మిత్రకు స్వాగతం!",
    message: "వ్యవసాయం సంబంధిత అన్ని ప్రశ్నల కోసం నేను మీ AI సహాయకుడిని. పంటలు, వాతావరణం, పురుగులు లేదా వ్యవసాయ పద్ధతుల గురించి నన్ను అడగండి. ప్రారంభించడానికి వాయిస్ ఇన్‌పుట్ లేదా త్వరిత ప్రశ్నలను ఉపయోగించండి!",
    button: "ప్రారంభించండి"
  },
  mr: {
    title: "किसानमित्र मध्ये आपले स्वागत आहे!",
    message: "मी शेतीशी संबंधित सर्व प्रश्नांसाठी तुमचा एआय सहाय्यक आहे. पिके, हवामान, कीटक किंवा शेती पद्धतींबद्दल मला विचारा. सुरुवात करण्यासाठी व्हॉइस इनपुट किंवा जलद प्रश्न वापरा!",
    button: "सुरु करा"
  },
  ta: {
    title: "கிஷன் மித்ராவில் வரவேற்கிறோம்!",
    message: "விவசாயம் தொடர்பான அனைத்து கேள்விகளுக்கும் நான் உங்கள் AI உதவியாளர். பயிர்கள், வானிலை, பூச்சிகள் அல்லது வேளாண்மை முறைகள் பற்றிச் என்னிடம் கேளுங்கள். தொடங்க குரல் உள்ளீடு அல்லது விரைவான கேள்விகளைப் பயன்படுத்தவும்!",
    button: "தொடங்கு"
  },
  gu: {
    title: "કિશનમિત્રમાં આપનું સ્વાગત છે!",
    message: "હું ખેતી સંબંધિત તમામ પ્રશ્નો માટે તમારો AI સહાયક છું. પાક, હવામાન, જીવાતો અથવા કૃષિ પદ્ધતિઓ વિશે મને પૂછો. પ્રારંભ કરવા માટે વોઇસ ઇનપુટ અથવા ઝડપી પ્રશ્નોનો ઉપયોગ કરો!",
    button: "શરૂ કરો"
  },
  ur: {
    title: "کسانمترہ میں خوش آمدید!",
    message: "میں کاشتکاری سے متعلق تمام سوالات کے لئے آپ کا اے آئی معاون ہوں۔ فصلوں، موسم، کیڑوں یا زرعی طریقوں کے بارے میں مجھ سے پوچھیں۔ شروع کرنے کے لئے وائس ان پٹ یا فوری سوالات استعمال کریں!",
    button: "شروع کریں"
  },
  kn: {
    title: "ಕೃಷಿಮಿತ್ರಕ್ಕೆ ಸ್ವಾಗತ!",
    message: "ಕೃಷಿ ಸಂಬಂಧಿತ ಎಲ್ಲಾ ಪ್ರಶ್ನೆಗಳಿಗೂ ನಾನು ನಿಮ್ಮ AI ಸಹಾಯಕನು. ಬೆಳೆಗಳು, ಹವಾಮಾನ, ಕೀಟಗಳು ಅಥವಾ ಕೃಷಿ ಪದ್ಧತಿಗಳ ಬಗ್ಗೆ ನನ್ನನ್ನು ಕೇಳಿ. ಪ್ರಾರಂಭಿಸಲು ಧ್ವನಿ ಇನ್‌ಪುಟ್ ಅಥವಾ ತ್ವರಿತ ಪ್ರಶ್ನೆಗಳನ್ನು ಬಳಸಿ!",
    button: "ಪ್ರಾರಂಭಿಸಿ"
  },
  or: {
    title: "କୃଷକମିତ୍ରକୁ ସ୍ୱାଗତ!",
    message: "ଚାଷ ସମ୍ବନ୍ଧୀୟ ସମସ୍ତ ପ୍ରଶ୍ନ ପାଇଁ ମୁଁ ଆପଣଙ୍କର AI ସହାୟକ। ଫସଲ, ପାଣିପାଗ, ପୋକାମାକଡ଼ କିମ୍ବା କୃଷି ପ୍ରଣାଳୀ ବିଷୟରେ ମତେ ପଚାରନ୍ତୁ। ଆରମ୍ଭ କରିବା ପାଇଁ ଭୋଇସ୍ ଇନପୁଟ କିମ୍ବା ତ୍ୱରିତ ପ୍ରଶ୍ନ ବ୍ୟବହାର କରନ୍ତୁ!",
    button: "ଆରମ୍ଭ କରନ୍ତୁ"
  },
  pa: {
    title: "ਕਿਸਾਨਮਿਤਰ ਵਿੱਚ ਤੁਹਾਡਾ ਸਵਾਗਤ ਹੈ!",
    message: "ਮੈਂ ਖੇਤੀ-ਬਾੜੀ ਨਾਲ ਸੰਬੰਧਿਤ ਸਾਰੇ ਪ੍ਰਸ਼ਨਾਂ ਲਈ ਤੁਹਾਡਾ AI ਸਹਾਇਕ ਹਾਂ। ਫਸਲਾਂ, ਮੌਸਮ, ਕੀੜੇ ਜਾਂ ਖੇਤੀਬਾੜੀ ਦੇ ਤਰੀਕਿਆਂ ਬਾਰੇ ਮੈਨੂੰ ਪੁੱਛੋ। ਸ਼ੁਰੂ ਕਰਨ ਲਈ ਵੌਇਸ ਇਨਪੁੱਟ ਜਾਂ ਤੇਜ਼ ਪ੍ਰਸ਼ਨ ਵਰਤੋਂ!",
    button: "ਸ਼ੁਰੂ ਕਰੋ"
  },
  ml: {
    title: "കർഷകമിത്രയിലേക്ക് സ്വാഗതം!",
    message: "കൃഷിയുമായി ബന്ധപ്പെട്ട എല്ലാ ചോദ്യങ്ങൾക്കും ഞാൻ നിങ്ങളുടെ AI സഹായി. വിളകൾ, കാലാവസ്ഥ, കീടങ്ങൾ അല്ലെങ്കിൽ കൃഷിരീതികൾ എന്നിവയെക്കുറിച്ച് എന്നോട് ചോദിക്കുക. തുടങ്ങാൻ വോയ്സ് ഇൻപുട്ട് അല്ലെങ്കിൽ ദ്രുത ചോദ്യങ്ങൾ ഉപയോഗിക്കുക!",
    button: "തുടങ്ങുക"
  },
  as: {
    title: "কৃষকমিত্ৰলৈ স্বাগতম!",
    message: "খেতি-বাটৰ সম্পৰ্কীয় সকলো প্ৰশ্নৰ বাবে মই আপোনাৰ AI সহায়ক। ফচল, বতাহ, পতংগ বা কৃষি প্ৰণালী সম্পৰ্কে মোক সোধক। আৰম্ভ কৰিবলৈ ভইচ ইনপুট বা তৎক্ষণাৎ প্ৰশ্ন ব্যৱহাৰ কৰক!",
    button: "আৰম্ভ কৰক"
  },
  mai: {
    title: "किसानमित्र में अहाँक स्वागत अछि!",
    message: "खेती संबंधी सब प्रश्नक लेल हम अहाँक AI सहायक छी। फसल, मौसम, कीड़ा या कृषि पद्धति संबंधी पूछू। शुरू करबाक लेल आवाज इनपुट या त्वरित प्रश्न प्रयोग करू!",
    button: "शुरू करू"
  },
  sat: {
    title: "ᱠᱤᱥᱟᱱᱢᱤᱴᱨᱟᱹ ᱨᱮ ᱥᱚᱹᱜᱮᱴ!",
    message: "ᱠᱷᱮᱛᱤ ᱡᱤᱱᱤᱞᱟᱹᱜ ᱡᱚᱱᱚᱜ ᱫᱚᱦᱚᱸ ᱢᱮ ᱟᱢᱟᱜ AI ᱥᱟᱭᱤᱠ ᱜᱟᱹᱛᱮᱭᱟᱹᱣ ᱠᱚᱜᱼᱟᱭᱟᱹ। ᱯᱚᱸᱛᱚᱸ, ᱥᱟᱹᱛᱷᱤ, ᱠᱤᱫᱚᱜ ᱟᱨᱮᱭᱟᱹᱱᱤ ᱯᱮᱛᱷᱟ ᱛᱮ ᱢᱮᱱᱟᱹᱢ ᱟᱹᱢ ᱥᱟᱹᱜᱮᱴ ᱚᱨᱭᱟᱢᱟᱹᱣ।",
    button: "ᱥᱟᱹᱜᱮᱴ"
  },
  kok: {
    title: "किसानमित्राक स्वागत!",
    message: "खेतराशी संबंदित सगळ्या प्रश्नाक लागीं हांव तुमका AI मदतगार. पिकां, हवामान, किडो वा शेती पद्धतींबाबत माझेर विचारात. सुरू करपाक आवाज इनपुट वा द्रुत प्रश्न वापरात!",
    button: "सुरू करात"
  },
  doi: {
    title: "किसानमित्र में आपदा स्वागत छ!",
    message: "खेती बाड़ी से जुड़्यां सगळें सवालां वास्ते मैं आपदा AI सहायक हां। फसल, मौसम, कीड़ा या खेती तरीक्यां बारे पुछो। शुरू करन वास्ते आवाज इनपुट या फटाफट सवाल करो!",
    button: "शुरू करो"
  },
  ne: {
    title: "किसानमित्रमा स्वागत छ!",
    message: "खेतीसम्बन्धी सबै प्रश्नहरूका लागि म तपाईंको AI सहायक हुँ। बाली, मौसम, किराहरू वा कृषि अभ्यासका बारेमा मसँग सोध्नुहोस्। सुरु गर्नका लागि आवाज इन्पुट वा छिटो प्रश्न प्रयोग गर्नुहोस्!",
    button: "सुरु गर्नुहोस्"
  },
  bo: {
    title: "ཀི་ཤཱན་མི་ཏྲའི་ལུ་ཕེབས་ཀྱི་མགོན་མོ!",
    message: "འབྲེལ་བ་ཡོད་པའི་ལས་རིགས་ཀྱི་དྲི་བ་ཚང་མའི་དོན་ལུ་ངའི་ཨའི་རིག་གནས་ལས་རོགས་རེད། སྲུང་བའི་ས་གནས། གནམ་གཤིས། འདུལ་བྱའི་དྲི་བའི་དོན་ལུ་ང་ལ་འདྲི། འགོ་བཙུགས་ནིའི་དོན་ལུ་སྐད་ཡིག་ཨིན་པུཊི་ཡང་ན་མགྱོགས་པའི་དྲི་བ་ལག་ལེན་འཐབ།",
    button: "འགོ་བཙུགས།"
  },
  sd: {
    title: "ڪسانمتر ۾ ڀليڪار!",
    message: "مان پوکي سان لاڳاپيل سڀني سوالن لاءِ توهان جو AI مددگار آهيان. فصلن، موسم، ڪِرن يا زرعي طريقيڪار بابت مون کان پڇو. شروع ڪرڻ لاءِ وائيس ان پٽ يا جلدي سوال استعمال ڪريو!",
    button: "شروع ڪريو"
  },
  ks: {
    title: "کسانمترس منز خوش آمدید!",
    message: "زراعت سٍ متعلق تمام سوالان ہندس جواب آسن چھ میون AI مددگار۔ فصل، موسم، کیڑا یا زرعی طریق کار بارے میہ پُتھ۔ شروعات کرن خٕطر وائس ان پُٹ یا تیز سوال استعمال کرن!",
    button: "شروع کرن"
  }
};


  const promptTemplates = {
  en: [
    "What crops grow best in my region?",
    "How to identify and treat tomato blight?",
    "When should I plant wheat?",
    "Sustainable pest control methods?",
    "Weather forecast for next week?"
  ],
  hi: [
    "मेरे क्षेत्र में कौन सी फसलें सबसे अच्छी होती हैं?",
    "टमाटर की झुलसा का पता और उपचार कैसे करें?",
    "गेहूं कब बोना चाहिए?",
    "टिकाऊ कीट नियंत्रण के तरीके?",
    "अगले सप्ताह के लिए मौसम का पूर्वानुमान?"
  ],
  bn: [
    "আমার অঞ্চলে কোন ফসল সবচেয়ে ভালো জন্মে?",
    "টমেটোর ব্লাইট চিহ্নিত ও চিকিৎসা কীভাবে করবেন?",
    "গম কখন রোপণ করা উচিত?",
    "টেকসই কীট নিয়ন্ত্রণ পদ্ধতি?",
    "আগামী সপ্তাহের আবহাওয়ার পূর্বাভাস কী?"
  ],
  te: [
    "నా ప్రాంతంలో ఏ పంటలు బాగా పెరుగుతాయి?",
    "టమాటో బ్లైట్‌ను ఎలా గుర్తించి చికిత్స చేయాలి?",
    "గోధుమ ఎప్పుడు నాటాలి?",
    "స్థిరమైన కీటక నియంత్రణ పద్ధతులు?",
    "తదుపరి వారానికి వాతావరణ సూచన?"
  ],
  mr: [
    "माझ्या प्रदेशात कोणती पिके सर्वाधिक चांगली वाढतात?",
    "टोमॅटो ब्लाइट ओळखून उपचार कसा करावा?",
    "गहू कधी पेरावा?",
    "शाश्वत कीड नियंत्रण पद्धती?",
    "पुढील आठवड्याचे हवामान अंदाज?"
  ],
  ta: [
    "என் பகுதியில் எந்த பயிர்கள் சிறப்பாக வளர்கின்றன?",
    "தக்காளி பிளைட்டை எவ்வாறு கண்டறிந்து சிகிச்சை செய்வது?",
    "கோதுமை எப்போது விதைக்க வேண்டும்?",
    "நிலையான பூச்சி கட்டுப்பாட்டு முறைகள்?",
    "அடுத்த வார வானிலை முன்னறிவிப்பு?"
  ],
  ur: [
    "میرے علاقے میں کون سی فصلیں سب سے بہتر اگتی ہیں؟",
    "ٹماٹر کے بلائٹ کی شناخت اور علاج کیسے کریں؟",
    "گندم کب کاشت کرنی چاہیے؟",
    "پائیدار کیڑوں کے کنٹرول کے طریقے؟",
    "اگلے ہفتے کا موسم کیسا ہوگا؟"
  ],
  gu: [
    "મારા વિસ્તારમાં કઈ ખેતી સારી થાય છે?",
    "ટમેટાની બ્લાઇટ ઓળખવી અને તેનું સારવાર કેવી રીતે કરવું?",
    "ગહું ક્યારે વાવવું?",
    "સતત કીટક નિયંત્રણ પદ્ધતિઓ?",
    "આવતા અઠવાડિયે હવામાનનું અનુમાન શું છે?"
  ],
  kn: [
    "ನನ್ನ ಪ್ರದೇಶದಲ್ಲಿ ಯಾವ ಬೆಳೆಗಳು ಉತ್ತಮವಾಗಿ ಬೆಳೆಯುತ್ತವೆ?",
    "ಟೊಮ್ಯಾಟೊ ಬ್ಲೈಟ್ ಅನ್ನು ಹೇಗೆ ಗುರುತಿಸಿ ಚಿಕಿತ್ಸೆ ಕೊಡಬೇಕು?",
    "ಗೋಧಿಯನ್ನು ಯಾವಾಗ ಬಿತ್ತಬೇಕು?",
    "ಸತತ ಕೀಟ ನಿಯಂತ್ರಣ ವಿಧಾನಗಳು?",
    "ಮುಂದಿನ ವಾರದ ಹವಾಮಾನ ಮುನ್ಸೂಚನೆ ಏನು?"
  ],
  ml: [
    "എന്റെ പ്രദേശത്ത് ഏത് വിളകളാണ് ഏറ്റവും നല്ലത്?",
    "തക്കാളി ബ്ലൈറ്റിനെ തിരിച്ചറിയാനും ചികിത്സിക്കാനും എങ്ങനെ?",
    "ഗോതമ്പ് എപ്പോഴാണ് നട്ടുപിടിപ്പിക്കേണ്ടത്?",
    "സ്ഥിരമായ കീടനിയന്ത്രണ രീതികൾ?",
    "അടുത്ത ആഴ്ചയിലെ കാലാവസ്ഥ പ്രവചനം?"
  ],
  or: [
    "ମୋ ଅଞ୍ଚଳରେ କେଉଁ ଫସଳ ସବୁଠାରୁ ଭଲ ହୁଏ?",
    "ଟମାଟୋ ବ୍ଲାଇଟ୍ କିପରି ପହିଚାରିବେ ଏବଂ ଚିକିତ୍ସା କରିବେ?",
    "ଗହମ କେବେ ପ୍ରତିରୋପଣ କରିବା ଉଚିତ?",
    "ସସ୍ଥାୟୀ କୀଟ ପରିଚାଳନା ପଦ୍ଧତି?",
    "ଆସନ୍ତା ସପ୍ତାହର ପାଣିପାଗ ପୂର୍ବାନୁମାନ?"
  ],
  pa: [
    "ਮੇਰੇ ਇਲਾਕੇ ਵਿੱਚ ਕਿਹੜੀਆਂ ਫਸਲਾਂ ਸਭ ਤੋਂ ਵਧੀਆ ਉੱਗਦੀਆਂ ਹਨ?",
    "ਟਮਾਟਰ ਬਲਾਈਟ ਨੂੰ ਕਿਵੇਂ ਪਛਾਣਣਾ ਅਤੇ ਇਲਾਜ ਕਰਨਾ ਹੈ?",
    "ਗੰਦਮ ਕਦੋਂ ਬੀਜਣੀ ਚਾਹੀਦੀ ਹੈ?",
    "ਟਿਕਾਊ ਕੀੜੇ ਕੰਟਰੋਲ ਤਰੀਕੇ?",
    "ਅਗਲੇ ਹਫ਼ਤੇ ਦਾ ਮੌਸਮ ਪੇਸ਼ਗੋਈ ਕੀ ਹੈ?"
  ],
  as: [
    "মোৰ অঞ্চলত কোনখন খেতি ভালকৈ হয়?",
    "টমেটো ব্লাইট চিনাক্ত আৰু চিকিৎসা কেনেকৈ কৰিব?",
    "গম কেতিয়া ল'ব লাগে?",
    "টেকসই পতঙ্গ নিয়ন্ত্ৰণ পদ্ধতি?",
    "আহি থকা সপ্তাহত বতৰ কিমান হ'ব?"
  ],
  mai: [
    "हमर इलाका में कौन फसल सबसँ नीक उगैत अछि?",
    "टमाटर ब्लाइट केँ पहचान कएनाइ आ उपचार कएनाइ कतय?",
    "गहुँ कखने बोअनाइ चाही?",
    "सतत कीट नियंत्रणक उपाय?",
    "आगाँ सप्ताहक मौसम पूर्वानुमान?"
  ],
  sat: [
    "आम इलाका रे काना फसल भल लागे?",
    "टमाटर ब्लाइट काना चिन्ह ओ इलाज कराय जाय?",
    "गेहूं काबे बोय जाय?",
    "सतत कीड़ा नियन्त्रन तरीक?",
    "आगाक सप्ताह रे मौसम खबर?"
  ],
  kok: [
    "माझ्या विभागांत कोणती पिकं चांगली वाढतात?",
    "टोमॅटो ब्लाइट कशे ओळखचे आ इलाज कशे करपाचो?",
    "गव्हाचेर कदां पेरपाचो?",
    "शाश्वत कीटक नियंत्रण पद्धती?",
    "पुढल्या आठवड्याचें हवामान अंदाज?"
  ],
  doi: [
    "मेरा क्षेत्रा में कन्ना फसल ठीक उगदा?",
    "टमाटर ब्लाइट किन्ना पचान्दा ते इलाज करदा?",
    "गहूं कदो बोना?",
    "टिकाऊ कीड़ा नियंत्रण तरीके?",
    "अगले हफ्ते का मौसम?"
  ],
  ne: [
    "मेरो क्षेत्रमा कुन बालीहरू राम्रो हुन्छन्?",
    "टमाटर ब्लाइट कसरी पहिचान गर्ने र उपचार गर्ने?",
    "गहुँ कहिले रोप्ने?",
    "दिगो किराहरू नियन्त्रण गर्ने तरिका?",
    "अर्को हप्ताको मौसम पूर्वानुमान?"
  ],
  sd: [
    "منهنجي علائقي ۾ ڪهڙا فصل بهتر ٿين ٿا؟",
    "ٽوماٽو بليٽ کي ڪيئن سڃاڻجي ۽ علاج ڪجي؟",
    "ڪڻڪ ڪڏهن پوکڻ گهرجي؟",
    "پائيدار ڀنگ ڪنٽرول طريقا؟",
    "اڳين هفتي جو موسم ڪهڙو ٿيندو؟"
  ],
  ks: [
    "مے زان پھتر كس فصل چُھ بہتر؟",
    "ٹماٹر بلیٹ كیسے پہچانوان تے علاج كریو؟",
    "گاہو كدنہ بون؟",
    "پائیدار کیڑا کنٹرول طریقے؟",
    "اگلے ہفتے موسم كہوا؟"
  ],
  brx: [
    "आंनि दानाय गावथां फसल दंफा?",
    "टमाटर ब्लाइट खौ नोमा आरो बिसारनाय ब्लाइ?",
    "गहमों माबाय बो?",
    "सस्टेनेबल कीड़ा खामानि फोर?",
    "फिनाय हाप्तायनि मौसम?"
  ],
  mni: [
    "Ei-mit amasung phaoba phajaba phajabani?",
    "Tomato blight kari oirabani amasung thadokpani?",
    "Wheat matam kada leirabani?",
    "Sustainable pest control phangba matam?",
    "Houjik haptagi weather forecast kari oirabani?"
  ],
  sa: [
    "मम प्रदेशे केनधान्यं श्रेष्ठं भवति?",
    "टमाटर-रोगस्य निदानं चिकित्सा च कथं?",
    "गोधूमः कदा रोपणीयः?",
    "सतत कीट-नियन्त्रण-पद्धतयः?",
    "आगामि सप्ताहस्य वृष्टि-पूर्वानुमानम्?"
  ]
};

  const handlePromptSelect = (template) => {
    setQuery(template);
    handleSubmit(null, template);
  };

  const backendUrl = "http://127.0.0.1:8000";



  useEffect(() => {
  if (user) {
    setShowWelcome(true);
  }
}, [user]);

  // Dark mode toggle
  useEffect(() => {
    document.body.className = darkMode ? "dark-mode" : "";
  }, [darkMode]);



  const handleLocationSelect = async () => {
    if (mapPosition) {
      // Update the location state with the selected map position
      const newLocation = { lat: mapPosition[0], lon: mapPosition[1] };
      setLocation(newLocation);
      
      try {
        const response = await axios.get(
          `https://nominatim.openstreetmap.org/reverse?format=json&lat=${mapPosition[0]}&lon=${mapPosition[1]}&zoom=10`
        );
        const address = response.data.address;
        const locationDisplay = address.village || address.town || address.city || address.county || address.state || 'Selected location';
        setLocationName(locationDisplay);
        
        // Optionally save the selected location to localStorage for persistence
        localStorage.setItem('kishanmitra_location', JSON.stringify(newLocation));
        localStorage.setItem('kishanmitra_location_name', locationDisplay);
        
        // Inform the user that location is updated (optional)
        setMessages(prev => [...prev, { 
          sender: "ai", 
          text: `Location updated to: ${locationDisplay}. Future responses will be tailored to this location.` 
        }]);
      } catch (error) {
        console.error("Error getting location name:", error);
        setLocationName("Selected location");
      }
      setShowLocationModal(false);
    }
  };
  const handleNewChat = () => {
    setMessages([]);
  };

  useEffect(() => {
    const onStorage = () => setUser(getUserFromToken());
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, []);

  useEffect(() => {
    const selectOrCreateChat = async () => {
      if (user && user.user_id) {
        const res = await axios.get(`${backendUrl}/user/chats`, { params: { user_id: user.user_id } });
        setChats(res.data.chats);

        if (res.data.chats && res.data.chats.length > 0) {
          setActiveChatId(res.data.chats[0].chat_id);
        } else {
          const formData = new FormData();
          formData.append("user_id", user.user_id);
          const createRes = await axios.post(`${backendUrl}/user/new_chat`, formData);
          setActiveChatId(createRes.data.chat_id);
          setChats([{ chat_id: createRes.data.chat_id, title: createRes.data.title || "New Chat" }]);
        }
      }
    };
    selectOrCreateChat();
  }, [user]);

  return (
  <>
      <ToastContainer /> 
    <Sidebar
      open={sidebarOpen}
      onToggle={toggleSidebar}
      user={user}
      chats={chats}
      activeChatId={activeChatId}
      setActiveChatId={setActiveChatId}
      refreshChats={refreshChats}
       onLogout={handleLogout}
    />
    <div
      className={`agri-bg ${darkMode ? "dark-mode" : ""}`}
      style={{
        marginLeft: sidebarOpen ? 220 : 0,
        transition: "margin-left 0.2s",
      }}
    >
      {/* Welcome Modal */}
      {showWelcome && (
        <div className="welcome-overlay">
          <div className="welcome-modal">
            <h2>{(welcomeContent[lang] || welcomeContent.en).title}</h2>
            <p>{(welcomeContent[lang] || welcomeContent.en).message}</p>
            <div className="welcome-features">
              <div className="feature">
                <span className="feature-icon">🗣️</span>
                <span>Voice Input</span>
              </div>
              <div className="feature">
                <span className="feature-icon">🌦️</span>
                <span>Weather Info</span>
              </div>
              <div className="feature">
                <span className="feature-icon">🌱</span>
                <span>Crop Advice</span>
              </div>
            </div>
            <button onClick={() => setShowWelcome(false)}>
              {(welcomeContent[lang] || welcomeContent.en).button}
            </button>
          </div>
        </div>
      )}

      {/* <button className="dark-toggle" onClick={() => setDarkMode(prev => !prev)}>
        {darkMode ? "🌞 Light Mode" : "🌙 Dark Mode"}
      </button> */}

      <div className="header">
        <img src={farmer} alt="Farm Logo" className="logo" />
        <h1>KishanMitra</h1>
        <p className="subtitle">Your smart companion for crops, weather & farming advice</p>
      </div>

      {/* Language and Location Selection */}
      <div className="control-panel">
        <div className="language-selector">
          <label htmlFor="language-select">Language:</label>
          <select
            id="language-select"
            value={lang}
            onChange={e => setLang(e.target.value)}
            style={{ marginBottom: 10, padding: 6, borderRadius: 6, width: "100%" }}
          >
            <option value="en">English</option>
            <option value="hi">हिन्दी (Hindi)</option>
            <option value="as">অসমীয়া (Assamese)</option>
            <option value="bn">বাংলা (Bengali)</option>
            <option value="brx">बोड़ो (Bodo)</option>
            <option value="doi">डोगरी (Dogri)</option>
            <option value="gu">ગુજરાતી (Gujarati)</option>
            <option value="kn">ಕನ್ನಡ (Kannada)</option>
            <option value="ks">کٲشُر (Kashmiri)</option>
            <option value="kok">कोंकणी (Konkani)</option>
            <option value="mai">मैथिली (Maithili)</option>
            <option value="ml">മലയാളം (Malayalam)</option>
            <option value="mni">মৈতৈলোন (Manipuri)</option>
            <option value="mr">मराठी (Marathi)</option>
            <option value="ne">नेपाली (Nepali)</option>
            <option value="or">ଓଡ଼ିଆ (Odia)</option>
            <option value="pa">ਪੰਜਾਬੀ (Punjabi)</option>
            <option value="sa">संस्कृत (Sanskrit)</option>
            <option value="sat">ᱥᱟᱱᱛᱟᱲᱤ (Santali)</option>
            <option value="sd">سنڌي (Sindhi)</option>
            <option value="ta">தமிழ் (Tamil)</option>
            <option value="te">తెలుగు (Telugu)</option>
            <option value="ur">اردو (Urdu)</option>
          </select>
        </div>

        <div className="location-selector">
          <label>Location:</label>
          <button
            className="location-btn"
            onClick={() => setShowLocationModal(true)}
            title="Set your location"
          >
            <FaMapMarkerAlt /> {locationName || formatLocation()}
          </button>
        </div>
      </div>

      {/* Location Modal */}
      {showLocationModal && (
        <div className="modal-overlay">
          <div className="location-modal">
            <h3>Select Your Location</h3>
            <form onSubmit={handleLocationSearch} className="location-search">
              <input
                type="text"
                placeholder="Enter city, state or address..."
                value={locationSearch}
                onChange={(e) => setLocationSearch(e.target.value)}
                className="location-search-input"
              />
              <button type="submit" className="location-search-btn">
                Search
              </button>
            </form>
            <p>Or click directly on the map to set your precise location</p>
            <div className="map-container">
              <MapContainer
                center={mapPosition}
                zoom={5}
                maxZoom={22}
                style={{ height: "50vh", width: "100%" }}
                whenCreated={(map) => { mapRef.current = map; }}
              >
                <TileLayer
                  attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                  url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                />
                <LocationMarker
                  position={mapPosition}
                  setPosition={setMapPosition}
                  mapRef={mapRef}
                />
              </MapContainer>
            </div>
            <div className="modal-actions">
              <button onClick={() => setShowLocationModal(false)}>Cancel</button>
              <button onClick={handleLocationSelect}>Confirm Location</button>
            </div>
          </div>
        </div>
      )}

      {/* Quick Prompt Templates */}
      <div className="prompt-templates">
        <h3>
          {lang === "en" ? "Quick Questions" :
            lang === "hi" ? "त्वरित प्रश्न" :
            lang === "bn" ? "দ্রুত প্রশ্ন" :
            lang === "te" ? "త్వరిత ప్రశ్నలు" :
            lang === "ta" ? "விரைவு கேள்விகள்" :
            lang === "mr" ? "जलद प्रश्न" :
            lang === "gu" ? "ઝડપી પ્રશ્નો" :
            lang === "kn" ? "ತ್ವರಿತ ಪ್ರಶ್ನೆಗಳು" :
            lang === "ml" ? "പെട്ടെന്ന് ചോദിക്കാവുന്ന ചോദ്യങ്ങൾ" :
            lang === "or" ? "ଦ୍ରୁତ ପ୍ରଶ୍ନଗୁଡ଼ିକ" :
            lang === "pa" ? "ਤੁਰੰਤ ਸਵਾਲ" :
            lang === "as" ? "দ্ৰুত প্ৰশ্নবোৰ" :
            lang === "mai" ? "शीघ्र प्रश्न" :
            lang === "sat" ? "तेज प्रश्न" :
            lang === "kok" ? "जलद प्रश्न" :
            lang === "doi" ? "झटपट सवाल" :
            lang === "ne" ? "छिटो प्रश्नहरू" :
            lang === "sd" ? "جلدي سوال" :
            lang === "ks" ? "ژر سوال" :
            lang === "brx" ? "गोजोनाय प्र्शन" :
            lang === "mni" ? "চৎকৃত প্ৰশ্ন" :
            lang === "sa" ? "त्वरितप्रश्नाः" :
            "Quick Questions"}
        </h3>
        <div className="template-container">
          {(promptTemplates[lang] || promptTemplates.en).map((template, index) => (
            <button
              key={index}
              className="template-button"
              onClick={() =>
                chatBoxRef.current &&
                chatBoxRef.current.sendPrompt &&
                chatBoxRef.current.sendPrompt(template)
              }
              disabled={isLoading || !activeChatId} // <-- disable if no chat id
            >
              {template}
            </button>
          ))}
        </div>
      </div>

      {/* ChatBox */}
      <div className="chat-container" style={{ maxWidth: 900, margin: "1px auto", width: "100%" }}>
        {activeChatId && (
          <ChatBox
            ref={chatBoxRef}
            activeChatId={activeChatId}
            user={user}
            refreshChats={refreshChats}
            location={location}
            lang={lang}
          />
        )}
        <footer className="footer">
          <span>🌾 Powered by AI for Farmers • {new Date().getFullYear()}</span>
        </footer>
      </div>
    </div>
  </>
);
}

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route path="/signup" element={<Signup />} />
        <Route path="/oauth-callback" element={<OAuthCallback />} />
        <Route path="/*" element={<MainApp />} />
      </Routes>
    </Router>
  );
}

export default App;
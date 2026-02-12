import React, { useState, useEffect, useRef, useMemo } from 'react';
import { createRoot } from 'react-dom/client';
import { GoogleGenAI, Type } from "@google/genai";
import * as XLSX_PKG from 'xlsx';
import { 
  Camera, 
  Save, 
  X, 
  Edit2, 
  Trash2, 
  Share2, 
  FileText, 
  Home, 
  List, 
  Settings, 
  ChevronLeft, 
  Check, 
  Search,
  Filter,
  Download
} from 'lucide-react';

// --- Configuration & Constants ---
const APP_NAME = "LeadCapture Pro";
const PRIMARY_COLOR = "bg-blue-900";
const ACCENT_COLOR = "text-teal-500";
const ACCENT_BG = "bg-teal-500";

// --- Library Normalization ---
// Handles differences between dev/prod builds where XLSX might be a default export or namespace
// @ts-ignore
const XLSX = XLSX_PKG.default ?? XLSX_PKG;

// --- Types ---
interface LeadEntity {
  id: string;
  companyName: string;
  address: string;
  contactPerson: string;
  contactNumber: string;
  whatsappNumber: string;
  email: string;
  website: string;
  natureOfBusiness: string;
  businessType: 'Trading' | 'Manufacturing' | 'Service' | 'Other';
  notes: string;
  scanDate: number; // Timestamp
}

type Screen = 'HOME' | 'SCAN' | 'EDIT' | 'LIST' | 'DETAIL' | 'SETTINGS';

// --- Services ---

// 1. Storage Service (Simulating Room DB)
const STORAGE_KEY = 'lead_capture_pro_db';

const LeadRepository = {
  getAll: (): LeadEntity[] => {
    try {
      const data = localStorage.getItem(STORAGE_KEY);
      return data ? JSON.parse(data) : [];
    } catch (e) {
      console.error("DB Error", e);
      return [];
    }
  },
  save: (lead: LeadEntity) => {
    const leads = LeadRepository.getAll();
    const existingIndex = leads.findIndex(l => l.id === lead.id);
    if (existingIndex >= 0) {
      leads[existingIndex] = lead;
    } else {
      leads.unshift(lead);
    }
    localStorage.setItem(STORAGE_KEY, JSON.stringify(leads));
  },
  delete: (id: string) => {
    const leads = LeadRepository.getAll().filter(l => l.id !== id);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(leads));
  },
  deleteAll: () => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify([]));
  }
};

// 1.1 Settings Service
const SETTINGS_KEY = 'lead_capture_pro_settings';

const SettingsRepository = {
  get: () => {
    try {
      const data = localStorage.getItem(SETTINGS_KEY);
      return data ? JSON.parse(data) : { autoSave: false };
    } catch (e) {
      return { autoSave: false };
    }
  },
  toggle: (key: string) => {
    const current = SettingsRepository.get();
    // @ts-ignore
    const newValue = !current[key];
    const newSettings = { ...current, [key]: newValue };
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(newSettings));
    return newSettings;
  }
};

// 2. Image Processing Service
const preprocessImage = (base64: string): Promise<string> => {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "Anonymous";
    img.onload = () => {
      try {
        const canvas = document.createElement('canvas');
        // Limit max dimension to maintain performance while keeping OCR quality
        const MAX_DIMENSION = 2560; // Increased to 2560px (approx 1440p+) to retain more detail
        let width = img.width;
        let height = img.height;
        
        if (width > MAX_DIMENSION || height > MAX_DIMENSION) {
           const ratio = Math.min(MAX_DIMENSION / width, MAX_DIMENSION / height);
           width = Math.round(width * ratio);
           height = Math.round(height * ratio);
        }

        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        if (!ctx) { resolve(base64); return; }

        ctx.drawImage(img, 0, 0, width, height);
        
        const imageData = ctx.getImageData(0, 0, width, height);
        const data = imageData.data;
        
        // Processing Parameters
        const contrast = 30; // 0-100 range for boost
        const factor = (259 * (contrast + 255)) / (255 * (259 - contrast));

        // PASS 1: Grayscale (Noise Reduction) + Contrast
        for (let i = 0; i < data.length; i += 4) {
          // Grayscale (Luma) - Eliminates chromatic noise which confuses OCR
          const luma = 0.299 * data[i] + 0.587 * data[i+1] + 0.114 * data[i+2];
          
          // Apply Contrast
          let cVal = factor * (luma - 128) + 128;
          
          // Clamp values
          if (cVal < 0) cVal = 0;
          if (cVal > 255) cVal = 255;
          
          data[i] = cVal;     // R
          data[i+1] = cVal;   // G
          data[i+2] = cVal;   // B
          // Alpha (data[i+3]) remains unchanged
        }

        // PASS 2: Sharpening (Convolution)
        // Kernel: 
        //  0 -1  0
        // -1  5 -1
        //  0 -1  0
        // This enhances edges for better character recognition
        
        const copy = new Uint8ClampedArray(data);
        
        for (let y = 1; y < height - 1; y++) {
          for (let x = 1; x < width - 1; x++) {
             const idx = (y * width + x) * 4;
             
             // Neighbor indices
             const up = ((y - 1) * width + x) * 4;
             const down = ((y + 1) * width + x) * 4;
             const left = (y * width + (x - 1)) * 4;
             const right = (y * width + (x + 1)) * 4;
             
             // Apply Kernel to one channel (since it's grayscale, R=G=B)
             const val = 5 * copy[idx] 
                       - copy[up] 
                       - copy[down] 
                       - copy[left] 
                       - copy[right];
             
             data[idx] = val;
             data[idx+1] = val;
             data[idx+2] = val;
          }
        }

        ctx.putImageData(imageData, 0, 0);
        resolve(canvas.toDataURL('image/jpeg', 0.9).split(',')[1]);
      } catch (e) {
        console.error("Image processing error", e);
        resolve(base64); // Fallback to original
      }
    };
    img.onerror = (e) => {
      console.error("Image load error", e);
      resolve(base64);
    };
    img.src = `data:image/jpeg;base64,${base64}`;
  });
};

// 3. AI Service (Gemini)
const processImageWithGemini = async (base64Image: string): Promise<Partial<LeadEntity>> => {
  try {
    const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
    
    // Pre-process image to enhance text visibility
    const processedImage = await preprocessImage(base64Image);
    
    const schema = {
      type: Type.OBJECT,
      properties: {
        companyName: { type: Type.STRING },
        address: { type: Type.STRING },
        contactPerson: { type: Type.STRING },
        contactNumber: { type: Type.STRING },
        whatsappNumber: { type: Type.STRING },
        email: { type: Type.STRING },
        website: { type: Type.STRING },
        natureOfBusiness: { type: Type.STRING },
        businessType: { type: Type.STRING, enum: ['Trading', 'Manufacturing', 'Service', 'Other'] },
        notes: { type: Type.STRING }
      },
      required: ['companyName', 'contactNumber']
    };

    const response = await ai.models.generateContent({
      model: 'gemini-3-flash-preview',
      contents: {
        parts: [
          {
            inlineData: {
              mimeType: 'image/jpeg',
              data: processedImage
            }
          },
          {
            text: `Extract business card details. Analyze carefully for small text. 
                   Format phone numbers cleanly. 
                   Detect business type based on keywords (e.g., 'Works' -> Manufacturing, 'Traders' -> Trading).
                   If a field is missing, return empty string.`
          }
        ]
      },
      config: {
        responseMimeType: 'application/json',
        responseSchema: schema
      }
    });

    const text = response.text;
    if (!text) throw new Error("No response from AI");
    
    return JSON.parse(text);
  } catch (error) {
    console.error("OCR Failed", error);
    throw error;
  }
};

// --- Components ---

const Header = ({ title, leftIcon, onLeftClick, rightIcon, onRightClick }: any) => (
  <div className={`${PRIMARY_COLOR} text-white p-4 pt-8 shadow-md flex items-center justify-between sticky top-0 z-50`}>
    <div className="flex items-center gap-3">
      {leftIcon && (
        <button onClick={onLeftClick} className="p-1 rounded-full hover:bg-white/10 transition">
          {leftIcon}
        </button>
      )}
      <h1 className="text-xl font-bold tracking-wide">{title}</h1>
    </div>
    {rightIcon && (
      <button onClick={onRightClick} className="p-1 rounded-full hover:bg-white/10 transition">
        {rightIcon}
      </button>
    )}
  </div>
);

const FAB = ({ onClick, icon }: any) => (
  <button 
    onClick={onClick}
    className={`fixed bottom-6 right-6 ${ACCENT_BG} text-white p-4 rounded-2xl shadow-lg hover:shadow-xl active:scale-95 transition-all z-50`}
  >
    {icon}
  </button>
);

const NavButton = ({ icon, label, active, onClick }: any) => (
  <button 
    onClick={onClick}
    className={`flex flex-col items-center justify-center w-full py-2 ${active ? 'text-blue-900' : 'text-gray-400'}`}
  >
    <div className={`mb-1 ${active ? 'scale-110' : ''} transition-transform`}>{icon}</div>
    <span className="text-[10px] font-medium">{label}</span>
  </button>
);

const BottomNav = ({ screen, setScreen }: any) => {
  if (screen === 'SCAN' || screen === 'EDIT') return null;
  return (
    <div className="fixed bottom-0 left-0 w-full bg-white border-t border-gray-200 flex justify-around pb-safe pt-2 z-40">
      <NavButton 
        icon={<Home size={24} />} 
        label="Home" 
        active={screen === 'HOME'} 
        onClick={() => setScreen('HOME')} 
      />
      <NavButton 
        icon={<List size={24} />} 
        label="My Leads" 
        active={screen === 'LIST'} 
        onClick={() => setScreen('LIST')} 
      />
      <NavButton 
        icon={<Settings size={24} />} 
        label="Settings" 
        active={screen === 'SETTINGS'} 
        onClick={() => setScreen('SETTINGS')} 
      />
    </div>
  );
};

// --- Screens ---

const HomeScreen = ({ leads, onScan }: any) => {
  const recentLeads = leads.slice(0, 3);
  const thisWeek = leads.filter((l: LeadEntity) => {
    const d = new Date(l.scanDate);
    const now = new Date();
    const diff = now.getTime() - d.getTime();
    return diff < 7 * 24 * 60 * 60 * 1000;
  }).length;

  return (
    <div className="pb-24 fade-in">
      <Header title={APP_NAME} />
      
      <div className="p-4 space-y-6">
        {/* Hero CTA */}
        <div className="bg-gradient-to-br from-blue-900 to-blue-800 rounded-3xl p-6 text-white shadow-lg relative overflow-hidden">
          <div className="absolute top-0 right-0 w-32 h-32 bg-white/5 rounded-full -mr-10 -mt-10"></div>
          <div className="relative z-10">
            <h2 className="text-2xl font-bold mb-2">Capture New Lead</h2>
            <p className="text-blue-200 mb-6 text-sm">Scan cards, banners, or signs instantly.</p>
            <button 
              onClick={onScan}
              className="bg-white text-blue-900 px-6 py-3 rounded-xl font-bold shadow-sm active:scale-95 transition flex items-center gap-2"
            >
              <Camera size={20} />
              Start Scanning
            </button>
          </div>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-white p-4 rounded-2xl shadow-sm border border-gray-100">
            <p className="text-gray-500 text-xs font-medium uppercase">Total Leads</p>
            <p className="text-3xl font-bold text-gray-900 mt-1">{leads.length}</p>
          </div>
          <div className="bg-white p-4 rounded-2xl shadow-sm border border-gray-100">
            <p className="text-gray-500 text-xs font-medium uppercase">This Week</p>
            <p className="text-3xl font-bold text-teal-600 mt-1">+{thisWeek}</p>
          </div>
        </div>

        {/* Recent */}
        <div>
          <div className="flex justify-between items-center mb-3">
            <h3 className="font-bold text-gray-800">Recent Activity</h3>
          </div>
          <div className="space-y-3">
            {recentLeads.length === 0 ? (
              <div className="text-center py-8 text-gray-400 bg-white rounded-xl border border-dashed border-gray-200">
                No leads yet
              </div>
            ) : (
              recentLeads.map((lead: LeadEntity) => (
                <div key={lead.id} className="bg-white p-4 rounded-xl shadow-sm border border-gray-100 flex items-center gap-4">
                  <div className={`w-10 h-10 rounded-full flex items-center justify-center text-white font-bold ${
                    lead.businessType === 'Manufacturing' ? 'bg-orange-500' :
                    lead.businessType === 'Trading' ? 'bg-indigo-500' : 'bg-teal-500'
                  }`}>
                    {lead.companyName.charAt(0).toUpperCase()}
                  </div>
                  <div className="flex-1 min-w-0">
                    <h4 className="font-semibold text-gray-900 truncate">{lead.companyName}</h4>
                    <p className="text-sm text-gray-500 truncate">{lead.contactPerson || 'Unknown Contact'}</p>
                  </div>
                  <span className="text-xs text-gray-400 whitespace-nowrap">
                    {new Date(lead.scanDate).toLocaleDateString()}
                  </span>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
      
      <FAB icon={<Camera />} onClick={onScan} />
    </div>
  );
};

const CameraScreen = ({ onCapture, onCancel }: any) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let stream: MediaStream;
    const startCamera = async () => {
      try {
        // Request higher resolution (4K ideal, usually settles for best available e.g. 1080p or 4K)
        stream = await navigator.mediaDevices.getUserMedia({ 
          video: { 
            facingMode: 'environment', 
            width: { ideal: 3840 }, 
            height: { ideal: 2160 } 
          } 
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        console.error("Camera access denied", err);
        alert("Camera permission is required to scan.");
        onCancel();
      }
    };
    startCamera();
    return () => {
      if (stream) stream.getTracks().forEach(track => track.stop());
    };
  }, []);

  const capture = () => {
    if (!videoRef.current) return;
    setLoading(true);
    const canvas = document.createElement('canvas');
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    canvas.getContext('2d')?.drawImage(videoRef.current, 0, 0);
    const base64 = canvas.toDataURL('image/jpeg', 0.8).split(',')[1];
    onCapture(base64);
  };

  return (
    <div className="fixed inset-0 bg-black z-50 flex flex-col">
      <div className="relative flex-1 bg-black overflow-hidden">
        <video ref={videoRef} autoPlay playsInline className="w-full h-full object-cover opacity-90" />
        {/* Overlay Guides */}
        <div className="absolute inset-0 border-2 border-white/30 m-8 rounded-xl pointer-events-none">
          <div className="absolute top-0 left-0 w-8 h-8 border-t-4 border-l-4 border-teal-500 -mt-1 -ml-1"></div>
          <div className="absolute top-0 right-0 w-8 h-8 border-t-4 border-r-4 border-teal-500 -mt-1 -mr-1"></div>
          <div className="absolute bottom-0 left-0 w-8 h-8 border-b-4 border-l-4 border-teal-500 -mb-1 -ml-1"></div>
          <div className="absolute bottom-0 right-0 w-8 h-8 border-b-4 border-r-4 border-teal-500 -mb-1 -mr-1"></div>
        </div>
        
        {loading && (
          <div className="absolute inset-0 bg-black/60 flex flex-col items-center justify-center text-white">
            <div className="w-12 h-12 border-4 border-teal-500 border-t-transparent rounded-full spin mb-4"></div>
            <p className="font-medium animate-pulse mt-4">Enhancing Image & Extracting Data...</p>
          </div>
        )}
      </div>

      <div className="h-32 bg-black flex items-center justify-around px-8">
        <button onClick={onCancel} className="p-4 rounded-full bg-white/10 text-white hover:bg-white/20">
          <X size={24} />
        </button>
        <button 
          onClick={capture}
          disabled={loading} 
          className="w-20 h-20 bg-white rounded-full border-4 border-gray-300 flex items-center justify-center active:scale-95 transition-transform"
        >
          <div className="w-16 h-16 bg-white border-2 border-black rounded-full"></div>
        </button>
        <div className="w-12"></div> {/* Spacer for balance */}
      </div>
    </div>
  );
};

const EditScreen = ({ initialData, onSave, onCancel }: any) => {
  const [formData, setFormData] = useState<LeadEntity>(initialData);

  const handleChange = (field: keyof LeadEntity, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  return (
    <div className="bg-gray-50 min-h-screen flex flex-col fade-in">
      <Header 
        title="Review & Save" 
        leftIcon={<ChevronLeft size={24} />} 
        onLeftClick={onCancel}
        rightIcon={<Save size={24} />}
        onRightClick={() => onSave(formData)}
      />
      
      <div className="p-4 pb-32 space-y-6">
        {/* Core Info Section */}
        <section className="bg-white p-5 rounded-2xl shadow-sm space-y-4">
          <h3 className="text-sm font-bold text-gray-400 uppercase tracking-wider mb-2">Business Details</h3>
          <div>
            <label className="text-xs text-gray-500 font-semibold ml-1">Company Name</label>
            <input 
              value={formData.companyName}
              onChange={e => handleChange('companyName', e.target.value)}
              className="w-full mt-1 p-3 bg-gray-50 border border-gray-200 rounded-xl focus:border-blue-500 focus:ring-2 focus:ring-blue-200 outline-none transition font-bold text-lg text-gray-800"
              placeholder="e.g. Acme Corp"
            />
          </div>
           <div>
            <label className="text-xs text-gray-500 font-semibold ml-1">Nature of Business</label>
            <input 
              value={formData.natureOfBusiness}
              onChange={e => handleChange('natureOfBusiness', e.target.value)}
              className="w-full mt-1 p-3 bg-gray-50 border border-gray-200 rounded-xl focus:border-blue-500 outline-none"
              placeholder="e.g. Steel Trading"
            />
          </div>
          <div className="grid grid-cols-2 gap-3">
             <div>
                <label className="text-xs text-gray-500 font-semibold ml-1">Type</label>
                <select 
                  value={formData.businessType}
                  onChange={e => handleChange('businessType', e.target.value as any)}
                  className="w-full mt-1 p-3 bg-gray-50 border border-gray-200 rounded-xl outline-none"
                >
                  <option value="Trading">Trading</option>
                  <option value="Manufacturing">Manufacturing</option>
                  <option value="Service">Service</option>
                  <option value="Other">Other</option>
                </select>
             </div>
          </div>
        </section>

        {/* Contact Section */}
        <section className="bg-white p-5 rounded-2xl shadow-sm space-y-4">
          <h3 className="text-sm font-bold text-gray-400 uppercase tracking-wider mb-2">Contact Info</h3>
          <div>
            <label className="text-xs text-gray-500 font-semibold ml-1">Contact Person</label>
            <input 
              value={formData.contactPerson}
              onChange={e => handleChange('contactPerson', e.target.value)}
              className="w-full mt-1 p-3 bg-gray-50 border border-gray-200 rounded-xl outline-none"
              placeholder="Full Name"
            />
          </div>
          <div className="grid grid-cols-1 gap-3">
            <div>
              <label className="text-xs text-gray-500 font-semibold ml-1">Phone Number</label>
              <input 
                type="tel"
                value={formData.contactNumber}
                onChange={e => handleChange('contactNumber', e.target.value)}
                className="w-full mt-1 p-3 bg-gray-50 border border-gray-200 rounded-xl outline-none"
                placeholder="+91..."
              />
            </div>
            <div>
              <label className="text-xs text-gray-500 font-semibold ml-1">Email</label>
              <input 
                type="email"
                value={formData.email}
                onChange={e => handleChange('email', e.target.value)}
                className="w-full mt-1 p-3 bg-gray-50 border border-gray-200 rounded-xl outline-none"
                placeholder="name@company.com"
              />
            </div>
          </div>
        </section>

        {/* Address Section */}
        <section className="bg-white p-5 rounded-2xl shadow-sm space-y-4">
           <h3 className="text-sm font-bold text-gray-400 uppercase tracking-wider mb-2">Location & Notes</h3>
           <div>
            <label className="text-xs text-gray-500 font-semibold ml-1">Address</label>
            <textarea 
              value={formData.address}
              onChange={e => handleChange('address', e.target.value)}
              className="w-full mt-1 p-3 bg-gray-50 border border-gray-200 rounded-xl outline-none h-24 resize-none"
              placeholder="Full Address..."
            />
          </div>
          <div>
            <label className="text-xs text-gray-500 font-semibold ml-1">Notes</label>
            <textarea 
              value={formData.notes}
              onChange={e => handleChange('notes', e.target.value)}
              className="w-full mt-1 p-3 bg-gray-50 border border-gray-200 rounded-xl outline-none h-20 resize-none"
              placeholder="Additional details..."
            />
          </div>
        </section>
      </div>
      
      {/* Floating Save Button (Redundant but UX friendly for long forms) */}
      <div className="fixed bottom-0 w-full bg-white p-4 border-t border-gray-200 z-40">
        <button 
          onClick={() => onSave(formData)}
          className="w-full bg-blue-900 text-white font-bold py-3 rounded-xl shadow-lg active:scale-95 transition"
        >
          Confirm & Save Lead
        </button>
      </div>
    </div>
  );
};

const ListScreen = ({ leads, onDelete, onEdit, onExport }: any) => {
  const [search, setSearch] = useState('');
  const [filterType, setFilterType] = useState('All');

  const filtered = leads.filter((l: LeadEntity) => {
    const matchSearch = l.companyName.toLowerCase().includes(search.toLowerCase()) || 
                        l.contactPerson.toLowerCase().includes(search.toLowerCase());
    const matchType = filterType === 'All' || l.businessType === filterType;
    return matchSearch && matchType;
  });

  const handleShare = async (lead: LeadEntity) => {
    const shareText = `
*${lead.companyName}*
----------------
👤 Contact: ${lead.contactPerson}
📞 Phone: ${lead.contactNumber}
📧 Email: ${lead.email}
📍 Address: ${lead.address}
🏢 Type: ${lead.businessType}
📝 Notes: ${lead.notes}
    `.trim();

    if (navigator.share) {
      try {
        await navigator.share({
          title: `Lead: ${lead.companyName}`,
          text: shareText,
        });
      } catch (err) {
        console.error('Error sharing', err);
      }
    } else {
      try {
        await navigator.clipboard.writeText(shareText);
        alert("Lead details copied to clipboard!");
      } catch (e) {
        alert("Sharing not supported on this device.");
      }
    }
  };

  return (
    <div className="bg-gray-50 min-h-screen pb-24 fade-in">
      <Header 
        title="My Leads" 
        rightIcon={<Download size={22} />}
        onRightClick={() => onExport(filtered)}
      />
      
      {/* Filters */}
      <div className="bg-white p-4 shadow-sm sticky top-[72px] z-30 space-y-3">
        <div className="relative">
          <Search className="absolute left-3 top-3 text-gray-400" size={18} />
          <input 
            className="w-full bg-gray-100 pl-10 pr-4 py-2.5 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 outline-none"
            placeholder="Search company or person..."
            value={search}
            onChange={e => setSearch(e.target.value)}
          />
        </div>
        <div className="flex gap-2 overflow-x-auto pb-1 no-scrollbar">
          {['All', 'Trading', 'Manufacturing', 'Service', 'Other'].map(type => (
            <button
              key={type}
              onClick={() => setFilterType(type)}
              className={`px-4 py-1.5 rounded-full text-xs font-medium whitespace-nowrap transition-colors ${
                filterType === type 
                  ? 'bg-blue-900 text-white' 
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              {type}
            </button>
          ))}
        </div>
      </div>

      <div className="p-4 space-y-3">
        {filtered.map((lead: LeadEntity) => (
          <div key={lead.id} className="bg-white rounded-xl p-4 shadow-sm border border-gray-100 relative group overflow-hidden">
            <div className="flex justify-between items-start mb-2">
              <div>
                <h3 className="font-bold text-gray-900">{lead.companyName}</h3>
                <p className="text-xs text-gray-500 font-medium bg-gray-100 inline-block px-2 py-0.5 rounded mt-1">
                  {lead.businessType}
                </p>
              </div>
              <div className="flex gap-2">
                 <button onClick={() => handleShare(lead)} className="p-2 text-gray-400 hover:text-green-600 bg-gray-50 rounded-lg">
                    <Share2 size={16} />
                 </button>
                 <button onClick={() => onEdit(lead)} className="p-2 text-gray-400 hover:text-blue-600 bg-gray-50 rounded-lg">
                    <Edit2 size={16} />
                 </button>
                 <button onClick={() => onDelete(lead.id)} className="p-2 text-gray-400 hover:text-red-500 bg-gray-50 rounded-lg">
                    <Trash2 size={16} />
                 </button>
              </div>
            </div>
            
            <div className="space-y-1 text-sm text-gray-600 mt-3">
              <div className="flex items-center gap-2">
                <span className="w-4 flex justify-center"><FileText size={14}/></span>
                <span className="truncate">{lead.contactPerson}</span>
              </div>
              {lead.contactNumber && (
                <div className="flex items-center gap-2">
                  <span className="w-4 flex justify-center text-green-600">📞</span>
                  <span className="font-mono">{lead.contactNumber}</span>
                </div>
              )}
            </div>
          </div>
        ))}
        
        {filtered.length === 0 && (
          <div className="text-center py-12 text-gray-400">
            <p>No leads found.</p>
          </div>
        )}
      </div>
    </div>
  );
};

const SettingsScreen = ({ onClear }: any) => {
  const [settings, setSettings] = useState(SettingsRepository.get());

  const toggleSetting = (key: string) => {
    const newSettings = SettingsRepository.toggle(key);
    setSettings(newSettings);
  };

  return (
    <div className="bg-gray-50 min-h-screen pb-24 fade-in">
      <Header title="Settings" />
      <div className="p-4 space-y-4">
        <div className="bg-white rounded-xl shadow-sm overflow-hidden">
          
          {/* Auto-Save Toggle */}
          <div 
            className="p-4 border-b border-gray-100 flex items-center justify-between cursor-pointer active:bg-gray-50 transition"
            onClick={() => toggleSetting('autoSave')}
          >
            <span className="font-medium text-gray-700">Auto-Save after Scan</span>
            <div className={`w-10 h-6 rounded-full relative transition-colors ${settings.autoSave ? 'bg-teal-500' : 'bg-gray-300'}`}>
              <div className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-all shadow-sm ${settings.autoSave ? 'right-1' : 'left-1'}`}></div>
            </div>
          </div>

          {/* Sound Toggle (Placeholder for visual consistency) */}
          <div className="p-4 border-b border-gray-100 flex items-center justify-between opacity-50">
             <span className="font-medium text-gray-700">Sound Effects</span>
             <div className="w-10 h-6 bg-gray-300 rounded-full relative"><div className="absolute left-1 top-1 w-4 h-4 bg-white rounded-full"></div></div>
          </div>
        </div>

        <button 
          onClick={() => {
            if (confirm("Are you sure you want to delete all data?")) onClear();
          }}
          className="w-full bg-white text-red-500 font-medium p-4 rounded-xl shadow-sm flex items-center justify-center gap-2 active:bg-red-50"
        >
          <Trash2 size={18} />
          Clear All Data
        </button>

        <div className="text-center text-xs text-gray-400 mt-8">
          <p>LeadCapture Pro v1.0.0</p>
          <p>Powered by Gemini AI</p>
        </div>
      </div>
    </div>
  );
};

// --- Main App Logic ---

const App = () => {
  const [screen, setScreen] = useState<Screen>('HOME');
  const [leads, setLeads] = useState<LeadEntity[]>([]);
  const [editingLead, setEditingLead] = useState<LeadEntity | null>(null);

  // Load leads on mount
  useEffect(() => {
    setLeads(LeadRepository.getAll());
  }, []);

  const handleCapture = async (base64: string) => {
    try {
      const extracted = await processImageWithGemini(base64);
      
      const newLead: LeadEntity = {
        id: Date.now().toString(),
        scanDate: Date.now(),
        companyName: extracted.companyName || '',
        address: extracted.address || '',
        contactPerson: extracted.contactPerson || '',
        contactNumber: extracted.contactNumber || '',
        whatsappNumber: extracted.whatsappNumber || '',
        email: extracted.email || '',
        website: extracted.website || '',
        natureOfBusiness: extracted.natureOfBusiness || '',
        businessType: (extracted.businessType as any) || 'Other',
        notes: extracted.notes || ''
      };

      // Check for auto-save setting
      const settings = SettingsRepository.get();
      if (settings.autoSave) {
         LeadRepository.save(newLead);
         setLeads(LeadRepository.getAll());
         setScreen('HOME');
      } else {
         setEditingLead(newLead);
         setScreen('EDIT');
      }
    } catch (e) {
      alert("Failed to extract text. Please try again.");
    }
  };

  const handleSaveLead = (lead: LeadEntity) => {
    LeadRepository.save(lead);
    setLeads(LeadRepository.getAll());
    setScreen('HOME');
    setEditingLead(null);
  };

  const handleDelete = (id: string) => {
    if (confirm("Delete this lead?")) {
      LeadRepository.delete(id);
      setLeads(LeadRepository.getAll());
    }
  };

  const handleExport = async (leadsToExport: LeadEntity[]) => {
    if (leadsToExport.length === 0) return alert("No leads to export.");

    const data = leadsToExport.map(l => ({
      "Company Name": l.companyName,
      "Address": l.address,
      "Contact Person": l.contactPerson,
      "Contact Number": l.contactNumber,
      "WhatsApp": l.whatsappNumber,
      "Email": l.email,
      "Website": l.website,
      "Nature of Business": l.natureOfBusiness,
      "Business Type": l.businessType,
      "Notes": l.notes,
      "Scan Date": new Date(l.scanDate).toLocaleString()
    }));

    const worksheet = XLSX.utils.json_to_sheet(data);
    
    // Auto column width (basic estimation)
    const colWidths = Object.keys(data[0]).map(key => ({
        wch: Math.max(key.length, ...data.map(row => (row[key as keyof typeof row] || "").toString().length)) + 2
    }));
    worksheet['!cols'] = colWidths;

    const workbook = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(workbook, worksheet, "Leads");

    const fileName = `Leads_Export_${new Date().toISOString().slice(0,10)}.xlsx`;

    // Try sharing first (mobile experience)
    if (navigator.canShare && navigator.share) {
         const wbout = XLSX.write(workbook, { bookType: 'xlsx', type: 'array' });
         const blob = new Blob([wbout], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' });
         const file = new File([blob], fileName, { type: blob.type });
         
         try {
             if (navigator.canShare({ files: [file] })) {
                 await navigator.share({
                     files: [file],
                     title: 'Exported Leads',
                     text: 'Here are the exported leads.'
                 });
                 return;
             }
         } catch (e) {
             console.log("Share failed, falling back to download", e);
         }
    }

    // Fallback to download
    XLSX.writeFile(workbook, fileName);
  };

  return (
    <div className="max-w-md mx-auto min-h-screen bg-white relative shadow-2xl overflow-hidden">
      {screen === 'HOME' && <HomeScreen leads={leads} onScan={() => setScreen('SCAN')} />}
      
      {screen === 'SCAN' && (
        <CameraScreen 
          onCapture={handleCapture} 
          onCancel={() => setScreen('HOME')} 
        />
      )}

      {screen === 'EDIT' && editingLead && (
        <EditScreen 
          initialData={editingLead}
          onSave={handleSaveLead}
          onCancel={() => {
            if (confirm("Discard changes?")) setScreen('HOME');
          }}
        />
      )}

      {screen === 'LIST' && (
        <ListScreen 
          leads={leads} 
          onDelete={handleDelete}
          onEdit={(lead: LeadEntity) => {
            setEditingLead(lead);
            setScreen('EDIT');
          }}
          onExport={handleExport}
        />
      )}

      {screen === 'SETTINGS' && (
        <SettingsScreen 
          onClear={() => {
            LeadRepository.deleteAll();
            setLeads([]);
            setScreen('HOME');
          }} 
        />
      )}

      <BottomNav screen={screen} setScreen={setScreen} />
    </div>
  );
};

const root = createRoot(document.getElementById('root')!);
root.render(<App />);
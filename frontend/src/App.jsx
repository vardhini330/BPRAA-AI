import { useState, useEffect, useRef } from 'react'
import { Upload, Droplet, Info, Clock, AlertTriangle, FileImage, Image as ImageIcon, ScanLine, Microscope, Target } from 'lucide-react'
import { checkHealth, getModelInfo, predictImage, getExplanation } from './services/api'

function App() {
  const [modelInfo, setModelInfo] = useState(null)
  const [health, setHealth] = useState(null)
  
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [result, setResult] = useState(null)
  
  const [explaining, setExplaining] = useState(false)
  const [explanation, setExplanation] = useState(null)
  
  const [history, setHistory] = useState([])
  
  const fileInputRef = useRef(null)

  // Load initial data and history
  useEffect(() => {
    const fetchSystemData = async () => {
      try {
        const [healthData, infoData] = await Promise.all([
          checkHealth(),
          getModelInfo()
        ])
        setHealth(healthData)
        setModelInfo(infoData)
      } catch (err) {
        console.error("Failed to connect to backend:", err)
        setError("Cannot connect to backend server. Ensure FastAPI is running.")
      }
    }
    
    fetchSystemData()
    
    const savedHistory = localStorage.getItem('bpraa_history')
    if (savedHistory) {
      try {
        setHistory(JSON.parse(savedHistory))
      } catch (e) {
        console.error("Could not parse history")
      }
    }
  }, [])

  // Save history
  useEffect(() => {
    localStorage.setItem('bpraa_history', JSON.stringify(history))
  }, [history])

  const handleFileSelect = (e) => {
    const file = e.target.files[0]
    if (!file) return
    
    if (!file.type.startsWith('image/')) {
      setError("Please select a valid image file.")
      return
    }

    setSelectedFile(file)
    setPreviewUrl(URL.createObjectURL(file))
    
    // Reset state for new image
    setResult(null)
    setExplanation(null)
    setError(null)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect({ target: { files: [e.dataTransfer.files[0]] } })
    }
  }

  const analyzeImage = async () => {
    if (!selectedFile) return
    
    setLoading(true)
    setError(null)
    setResult(null)
    setExplanation(null)
    
    try {
      const data = await predictImage(selectedFile)
      setResult(data)
      
      const historyEntry = {
        id: Date.now(),
        filename: selectedFile.name,
        prediction: data.prediction,
        confidence: data.confidence,
        timestamp: new Date().toISOString()
      }
      setHistory(prev => [historyEntry, ...prev].slice(0, 10))
      
    } catch (err) {
      setError(err.response?.data?.detail || "An error occurred during prediction.")
    } finally {
      setLoading(false)
    }
  }

  const generateExplanation = async () => {
    if (!selectedFile) return
    
    setExplaining(true)
    setError(null)
    
    try {
      const data = await getExplanation(selectedFile)
      setExplanation(data)
    } catch (err) {
      setError(err.response?.data?.detail || "Failed to generate Grad-CAM explanation.")
    } finally {
      setExplaining(false)
    }
  }

  const formatConfidence = (val) => `${(val * 100).toFixed(2)}%`

  return (
    <div className="min-h-screen bg-slate-900 text-slate-200 font-sans pb-12 relative overflow-hidden">
      {/* Forensic Grid Background */}
      <div className="absolute inset-0 z-0 opacity-10 pointer-events-none" 
           style={{ backgroundImage: 'linear-gradient(to right, #475569 1px, transparent 1px), linear-gradient(to bottom, #475569 1px, transparent 1px)', backgroundSize: '40px 40px' }}>
      </div>

      {/* Header */}
      <header className="bg-slate-950 border-b-4 border-rose-800 shadow-xl relative z-10">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="bg-rose-900 p-2 rounded-lg border border-rose-700 shadow-inner">
              <Droplet className="text-rose-400 h-6 w-6" />
            </div>
            <div>
              <h1 className="font-bold text-2xl tracking-widest text-slate-100 uppercase font-mono">
                BPRAA
              </h1>
              <span className="hidden sm:block text-rose-500 text-xs font-semibold tracking-wider uppercase mt-0.5">
                Forensic Bloodstain Analysis Terminal
              </span>
            </div>
          </div>
          
          <div className="flex items-center text-sm font-mono border border-slate-800 rounded-md overflow-hidden bg-slate-900">
            <div className="px-3 py-2 bg-slate-800 text-slate-400 border-r border-slate-700">SYSTEM</div>
            {health?.status === 'healthy' ? (
              <span className="flex items-center text-emerald-400 px-4 py-2 font-medium tracking-wide">
                <span className="w-2 h-2 rounded-full bg-emerald-500 mr-2 shadow-[0_0_8px_rgba(16,185,129,0.8)]"></span>
                ONLINE
              </span>
            ) : (
              <span className="flex items-center text-amber-500 px-4 py-2 font-medium tracking-wide">
                <span className="w-2 h-2 rounded-full bg-amber-500 mr-2 animate-pulse shadow-[0_0_8px_rgba(245,158,11,0.8)]"></span>
                CONNECTING
              </span>
            )}
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 mt-8 grid grid-cols-1 lg:grid-cols-12 gap-6 relative z-10">
        
        {/* Left Column (Main functionality) */}
        <div className="lg:col-span-8 space-y-6">
          
          {/* Disclaimer */}
          <div className="bg-slate-800/80 backdrop-blur-sm border-l-4 border-rose-600 rounded-r-lg p-4 flex gap-3 text-slate-300 shadow-lg">
            <Info className="shrink-0 mt-0.5 h-5 w-5 text-rose-500" />
            <p className="text-sm leading-relaxed">
              <strong className="text-rose-400 font-mono tracking-wide">CLASSIFIED PROTOCOL:</strong> BPRAA is an AI-assisted research prototype. 
              Predictions are intended for research and educational purposes and must not replace forensic examination by qualified professionals.
            </p>
          </div>

          {error && (
            <div className="bg-red-900/40 border border-red-500/50 rounded-lg p-4 flex gap-3 text-red-200 shadow-lg">
              <AlertTriangle className="shrink-0 mt-0.5 h-5 w-5 text-red-400" />
              <p className="text-sm font-mono">{error}</p>
            </div>
          )}

          {/* Upload Area */}
          <div className="bg-slate-900 border border-slate-700 rounded-xl shadow-2xl overflow-hidden">
            <div className="bg-slate-800/50 px-6 py-3 border-b border-slate-700 flex items-center">
              <ScanLine className="w-4 h-4 mr-2 text-rose-500" />
              <h2 className="text-sm font-mono font-semibold text-slate-300 uppercase tracking-widest">Evidence Input Terminal</h2>
            </div>
            
            <div className="p-6">
              <div 
                className={`border-2 border-dashed rounded-lg p-8 text-center transition-all cursor-pointer bg-slate-950/50
                  ${previewUrl ? 'border-slate-700' : 'border-slate-600 hover:border-rose-500 hover:bg-slate-800/50'}`}
                onDragOver={(e) => e.preventDefault()}
                onDrop={handleDrop}
                onClick={() => !previewUrl && fileInputRef.current?.click()}
              >
                <input 
                  type="file" 
                  ref={fileInputRef} 
                  onChange={handleFileSelect} 
                  accept="image/*"
                  className="hidden" 
                />
                
                {!previewUrl ? (
                  <div className="flex flex-col items-center">
                    <div className="p-4 bg-slate-800 rounded-full shadow-inner mb-4 border border-slate-700">
                      <Upload className="h-8 w-8 text-slate-400 group-hover:text-rose-500 transition-colors" />
                    </div>
                    <p className="font-mono text-slate-300 tracking-wide">INITIALIZE SECURE UPLOAD</p>
                    <p className="text-xs font-mono text-slate-500 mt-2 tracking-widest">DRAG & DROP EVIDENCE OR CLICK TO BROWSE</p>
                  </div>
                ) : (
                  <div className="flex flex-col sm:flex-row items-center justify-center gap-8">
                    <div className="relative group">
                      <div className="absolute inset-0 border-2 border-rose-500/0 group-hover:border-rose-500/50 transition-colors rounded-lg pointer-events-none"></div>
                      <img 
                        src={previewUrl} 
                        alt="Evidence Preview" 
                        className="max-h-64 rounded-lg shadow-2xl border border-slate-600 object-contain bg-black"
                      />
                      <button 
                        onClick={(e) => { e.stopPropagation(); setPreviewUrl(null); setSelectedFile(null); setResult(null); setExplanation(null); }}
                        className="absolute -top-3 -right-3 bg-slate-800 text-slate-400 hover:text-rose-500 hover:border-rose-500 rounded-full p-1.5 shadow-xl border border-slate-600 transition-colors"
                      >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"></path></svg>
                      </button>
                    </div>
                    <div className="flex flex-col gap-3 min-w-[200px]">
                      <button
                        onClick={(e) => { e.stopPropagation(); analyzeImage(); }}
                        disabled={loading}
                        className="w-full py-3 px-4 bg-rose-700 hover:bg-rose-600 disabled:bg-slate-700 disabled:text-slate-500 text-white font-mono font-bold tracking-wider rounded border border-rose-800 hover:border-rose-500 hover:shadow-[0_0_15px_rgba(225,29,72,0.4)] transition-all flex items-center justify-center"
                      >
                        {loading ? (
                          <><span className="animate-spin h-4 w-4 mr-3 border-2 border-slate-400 border-t-white rounded-full"></span> PROCESSING...</>
                        ) : (
                          <><Microscope className="w-5 h-5 mr-2" /> EXECUTE ANALYSIS</>
                        )}
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Results Area */}
          {result && (
            <div className="bg-slate-900 border border-emerald-900/50 rounded-xl shadow-2xl overflow-hidden animate-in fade-in slide-in-from-bottom-4 duration-500 relative">
              <div className="absolute top-0 left-0 w-1 h-full bg-emerald-600"></div>
              <div className="bg-slate-800/80 px-6 py-3 border-b border-slate-700 flex items-center justify-between">
                <div className="flex items-center">
                  <Target className="w-4 h-4 mr-2 text-emerald-500" />
                  <h3 className="text-sm font-mono font-semibold text-slate-200 uppercase tracking-widest">
                    Analysis Report Generated
                  </h3>
                </div>
                <span className="text-xs font-mono text-emerald-500 border border-emerald-900 px-2 py-0.5 rounded bg-emerald-950/30">MATCH DETECTED</span>
              </div>
              
              <div className="p-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                  {/* Primary Result */}
                  <div className="flex flex-col justify-center items-center p-6 bg-slate-950 rounded-lg border border-slate-800 shadow-inner">
                    <span className="text-slate-500 font-mono text-xs tracking-widest mb-2 uppercase">Predicted Origin Height</span>
                    <span className="text-5xl font-black text-white tracking-tighter drop-shadow-[0_0_10px_rgba(255,255,255,0.2)]">
                      {result.prediction}
                    </span>
                    <span className="mt-4 text-xs font-mono px-3 py-1 bg-slate-800 border border-slate-700 rounded text-emerald-400 tracking-wider">
                      CONFIDENCE: {formatConfidence(result.confidence)}
                    </span>
                  </div>
                  
                  {/* Probability Breakdown */}
                  <div className="flex flex-col justify-center pl-4 border-l border-slate-800">
                    <h4 className="text-xs font-mono font-semibold text-slate-500 uppercase tracking-widest mb-4">Probability Metrics</h4>
                    <div className="space-y-5">
                      {Object.entries(result.probabilities).map(([className, prob]) => (
                        <div key={className}>
                          <div className="flex justify-between text-sm mb-1.5 font-mono">
                            <span className="text-slate-300">{className}</span>
                            <span className={className === result.prediction ? 'text-emerald-400 font-bold' : 'text-slate-500'}>
                              {formatConfidence(prob)}
                            </span>
                          </div>
                          <div className="w-full bg-slate-800 rounded-sm h-1.5 overflow-hidden">
                            <div 
                              className={`h-full rounded-sm transition-all duration-1000 ${className === result.prediction ? 'bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.8)]' : 'bg-slate-600'}`} 
                              style={{ width: `${Math.max(prob * 100, 2)}%` }}
                            ></div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
                
                {/* Explain Button */}
                <div className="mt-8 pt-6 border-t border-slate-800 text-center">
                  {!explanation ? (
                    <button
                      onClick={generateExplanation}
                      disabled={explaining}
                      className="inline-flex items-center px-6 py-2.5 bg-slate-800 border border-slate-600 hover:border-cyan-500 hover:text-cyan-400 text-slate-300 font-mono text-sm tracking-wide rounded shadow-lg transition-all"
                    >
                      {explaining ? (
                        <><span className="animate-spin h-4 w-4 mr-3 border-2 border-cyan-600 border-t-transparent rounded-full"></span> RUNNING DEEP SCAN...</>
                      ) : (
                        <><ScanLine className="w-4 h-4 mr-2" /> GENERATE GRAD-CAM OVERLAY</>
                      )}
                    </button>
                  ) : (
                    <div className="space-y-4 animate-in fade-in zoom-in-95 duration-500 p-4 border border-cyan-900/50 bg-slate-950 rounded-lg">
                      <h4 className="font-mono text-cyan-500 text-sm tracking-widest uppercase flex items-center justify-center pb-2 border-b border-slate-800">
                        <ImageIcon className="w-4 h-4 mr-2" />
                        Attention Mapping Active
                      </h4>
                      <div className="inline-block border-2 border-slate-700 p-1 bg-black rounded shadow-2xl relative group">
                         <div className="absolute inset-0 border border-cyan-500/20 pointer-events-none z-10"></div>
                         <img 
                            src={`data:image/png;base64,${explanation.gradcam_base64}`} 
                            alt="Grad-CAM overlay"
                            className="max-w-full rounded"
                            style={{ maxHeight: '400px' }}
                         />
                         {/* Target reticle overlay for style */}
                         <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-12 h-12 border border-cyan-500/30 rounded-full pointer-events-none"></div>
                         <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-2 h-2 bg-cyan-500/50 rounded-full pointer-events-none"></div>
                      </div>
                      <p className="text-xs font-mono text-slate-500 bg-slate-900 p-3 rounded border border-slate-800 text-left mt-4">
                        <AlertTriangle className="inline-block w-3 h-3 mr-1 -mt-0.5 text-amber-500" />
                        {explanation.disclaimer} Layer: <span className="text-slate-400">{explanation.target_layer}</span>
                      </p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

        </div>
        
        {/* Right Column (Info & History) */}
        <div className="lg:col-span-4 space-y-6">
          
          {/* Model Info */}
          <div className="bg-slate-900 border border-slate-700 rounded-xl shadow-xl overflow-hidden">
             <div className="bg-slate-800/50 px-4 py-3 border-b border-slate-700 flex items-center">
              <Info className="w-4 h-4 mr-2 text-slate-400" />
              <h3 className="font-mono text-sm text-slate-300 uppercase tracking-widest">Model Specifications</h3>
            </div>
            
            <div className="p-5">
              {modelInfo ? (
                <dl className="space-y-4 font-mono text-xs">
                  <div className="flex flex-col border-b border-slate-800 pb-2">
                    <dt className="text-slate-500 uppercase tracking-wider mb-1">Architecture</dt>
                    <dd className="text-rose-400 font-semibold">{modelInfo.model_name} <span className="text-slate-400 font-normal">({modelInfo.architecture})</span></dd>
                  </div>
                  <div className="flex flex-col border-b border-slate-800 pb-2">
                    <dt className="text-slate-500 uppercase tracking-wider mb-1">Task Protocol</dt>
                    <dd className="text-slate-300">{modelInfo.task}</dd>
                  </div>
                  <div className="flex flex-col border-b border-slate-800 pb-2">
                    <dt className="text-slate-500 uppercase tracking-wider mb-1">Target Classes</dt>
                    <dd className="flex gap-2">
                      {modelInfo.classes.map(c => <span key={c} className="bg-slate-800 px-2 py-0.5 rounded text-slate-300 border border-slate-700">{c}</span>)}
                    </dd>
                  </div>
                  <div className="flex flex-col border-b border-slate-800 pb-2">
                    <dt className="text-slate-500 uppercase tracking-wider mb-1">Input Matrix</dt>
                    <dd className="text-slate-300">{modelInfo.input_size}</dd>
                  </div>
                  <div className="flex flex-col pt-1">
                    <dt className="text-slate-500 uppercase tracking-wider mb-1">Engine</dt>
                    <dd className="text-emerald-400 flex items-center">
                      <span className="w-1.5 h-1.5 bg-emerald-500 rounded-full mr-2"></span>
                      PyTorch ({modelInfo.device.toUpperCase()})
                    </dd>
                  </div>
                </dl>
              ) : (
                <div className="text-sm font-mono text-slate-600 animate-pulse flex items-center">
                  <span className="w-2 h-2 bg-slate-600 rounded-full mr-2 animate-ping"></span>
                  FETCHING SPECS...
                </div>
              )}
            </div>
          </div>

          {/* History */}
          <div className="bg-slate-900 border border-slate-700 rounded-xl shadow-xl overflow-hidden">
            <div className="bg-slate-800/50 px-4 py-3 border-b border-slate-700 flex items-center">
              <Clock className="w-4 h-4 mr-2 text-slate-400" />
              <h3 className="font-mono text-sm text-slate-300 uppercase tracking-widest">Case History</h3>
            </div>
            
            <div className="p-3">
              {history.length > 0 ? (
                <div className="space-y-1">
                  {history.map((item) => (
                    <div key={item.id} className="flex justify-between items-center text-xs font-mono p-2 hover:bg-slate-800 rounded border border-transparent hover:border-slate-700 transition-colors group">
                      <div className="truncate pr-3">
                        <p className="text-slate-300 truncate" title={item.filename}>{item.filename}</p>
                        <p className="text-slate-600 mt-0.5">{new Date(item.timestamp).toLocaleTimeString()}</p>
                      </div>
                      <div className="text-right shrink-0 bg-slate-950 px-2 py-1 rounded border border-slate-800">
                        <p className="font-bold text-slate-300 group-hover:text-emerald-400 transition-colors">{item.prediction}</p>
                        <p className="text-slate-500 text-[10px]">{formatConfidence(item.confidence)}</p>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-xs font-mono text-slate-600 text-center py-8 border border-dashed border-slate-800 rounded mx-2 my-2">
                  NO LOGS FOUND
                </div>
              )}
            </div>
          </div>

        </div>
      </main>
    </div>
  )
}

export default App

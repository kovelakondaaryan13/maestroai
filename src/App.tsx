/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { GoogleGenAI, Modality, ThinkingLevel, LiveServerMessage } from "@google/genai";
import { motion, AnimatePresence } from "motion/react";
import { 
  Music, 
  Mic, 
  Guitar, 
  Sparkles, 
  CheckCircle2, 
  XCircle,
  AlertCircle,
  Play,
  RotateCcw,
  ArrowRight,
  Lock,
  LogOut,
  Phone,
  Chrome,
  Activity,
  Trophy,
  Zap,
  MessageSquare,
  MicOff,
  Volume2
} from "lucide-react";
import Markdown from 'react-markdown';

// Initialize Gemini
const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

// --- Types ---

type Track = 'Singing' | 'Guitar';
type View = 'auth' | 'track_select' | 'calibration' | 'test_select' | 'pitch_test' | 'interval_test' | 'stability_test' | 'report' | 'live_session';

interface VocalRange {
  low: number;
  high: number;
}

interface PitchResult {
  note: string;
  targetFreq: number;
  detectedFreq: number | null;
  accuracy: number; // 0 to 1
  status: 'green' | 'yellow' | 'red' | 'none';
  detail?: string;
}

const SINGING_NOTES = [
  { name: 'Sa', note: 'C4', freq: 261.63 },
  { name: 'Re', note: 'D4', freq: 293.66 },
  { name: 'Ga', note: 'E4', freq: 329.63 },
  { name: 'Ma', note: 'F4', freq: 349.23 },
  { name: 'Pa', note: 'G4', freq: 392.00 },
  { name: 'Da', note: 'A4', freq: 440.00 },
  { name: 'Ni', note: 'B4', freq: 493.88 },
  { name: 'Sa\'', note: 'C5', freq: 523.25 },
];

const GUITAR_NOTES = [
  { name: 'E', note: 'E2', freq: 82.41 },
  { name: 'A', note: 'A2', freq: 110.00 },
  { name: 'D', note: 'D3', freq: 146.83 },
  { name: 'G', note: 'G3', freq: 196.00 },
  { name: 'B', note: 'B3', freq: 246.94 },
  { name: 'e', note: 'E4', freq: 329.63 },
];

// --- Pitch Detection Logic ---

function autoCorrelate(buffer: Float32Array, sampleRate: number) {
  const size = buffer.length;
  let rms = 0;

  for (let i = 0; i < size; i++) {
    const val = buffer[i];
    rms += val * val;
  }
  rms = Math.sqrt(rms / size);
  if (rms < 0.005) return { pitch: -1, confidence: 0 };

  const clipped = new Float32Array(size);
  const clipLimit = rms * 0.5;
  for (let i = 0; i < size; i++) {
    if (Math.abs(buffer[i]) > clipLimit) {
      clipped[i] = buffer[i] > 0 ? buffer[i] - clipLimit : buffer[i] + clipLimit;
    } else {
      clipped[i] = 0;
    }
  }

  const c = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size - i; j++) {
      c[i] = c[i] + clipped[j] * clipped[j + i];
    }
  }

  let d = 0;
  while (c[d] > c[d + 1]) d++;
  
  let maxval = -1, maxpos = -1;
  for (let i = d; i < size; i++) {
    if (c[i] > maxval) {
      maxval = c[i];
      maxpos = i;
    }
  }

  const confidence = maxval / c[0];
  if (maxpos === -1 || confidence < 0.2) return { pitch: -1, confidence: 0 };

  let T0 = maxpos;
  if (T0 > 0 && T0 < size - 1) {
    const x1 = c[T0 - 1], x2 = c[T0], x3 = c[T0 + 1];
    const a = (x1 + x3 - 2 * x2) / 2;
    const b = (x3 - x1) / 2;
    if (a) T0 = T0 - b / (2 * a);
  }

  return { pitch: sampleRate / T0, confidence };
}

// --- Components ---

const VibeBackground = ({ amplitude }: { amplitude: number }) => (
  <div className="fixed inset-0 -z-10 overflow-hidden pointer-events-none">
    <motion.div 
      animate={{ 
        scale: 1 + amplitude * 3,
        opacity: 0.1 + amplitude * 0.4
      }}
      className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[1000px] h-[1000px] bg-studio-accent/10 rounded-full blur-[150px]"
    />
    <div className="absolute inset-0 bg-studio-bg/80 backdrop-blur-[1px]" />
  </div>
);

const PitchGraph = ({ currentPitch, targetFreq, isRecording }: { currentPitch: number | null, targetFreq: number, isRecording: boolean }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const history = useRef<{ time: number, pitch: number | null }[]>([]);
  const lastTime = useRef(Date.now());

  useEffect(() => {
    if (!isRecording) {
      history.current = [];
      return;
    }

    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const render = () => {
      const now = Date.now();
      lastTime.current = now;

      // Add current pitch to history
      history.current.push({ time: now, pitch: currentPitch });
      // Keep only last 5 seconds
      history.current = history.current.filter(h => now - h.time < 5000);

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      // Draw target line
      const targetY = canvas.height / 2;
      ctx.beginPath();
      ctx.setLineDash([5, 5]);
      ctx.strokeStyle = 'rgba(197, 160, 89, 0.2)';
      ctx.moveTo(0, targetY);
      ctx.lineTo(canvas.width, targetY);
      ctx.stroke();
      ctx.setLineDash([]);

      // Draw pitch history
      if (history.current.length > 1) {
        ctx.lineWidth = 3;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        
        let lastX = -1, lastY = -1;

        history.current.forEach((h, i) => {
          if (i === 0) return;
          const prev = history.current[i-1];
          
          const x2 = canvas.width - (now - h.time) / 5000 * canvas.width;
          if (h.pitch === null) {
            lastX = -1;
            return;
          }
          
          const semitoneDiff2 = 12 * Math.log2(h.pitch / targetFreq);
          const y2 = targetY - (semitoneDiff2 * 30);

          if (prev.pitch !== null) {
            const x1 = canvas.width - (now - prev.time) / 5000 * canvas.width;
            const semitoneDiff1 = 12 * Math.log2(prev.pitch / targetFreq);
            const y1 = targetY - (semitoneDiff1 * 30);

            const accuracy = Math.max(0, 1 - Math.abs(semitoneDiff2));
            ctx.strokeStyle = accuracy > 0.9 ? '#C5A059' : accuracy > 0.7 ? '#A67C37' : '#991B1B';
            
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.stroke();
          }
          
          lastX = x2;
          lastY = y2;
        });

        // Draw current point
        if (lastX !== -1) {
          ctx.beginPath();
          ctx.arc(lastX, lastY, 6, 0, Math.PI * 2);
          ctx.fillStyle = ctx.strokeStyle;
          ctx.fill();
          ctx.strokeStyle = '#fff';
          ctx.lineWidth = 2;
          ctx.stroke();
        }
      }

      if (isRecording) requestAnimationFrame(render);
    };

    const anim = requestAnimationFrame(render);
    return () => cancelAnimationFrame(anim);
  }, [isRecording, currentPitch, targetFreq]);

  return (
    <div className="glass-card p-6 h-64 relative overflow-hidden">
      <div className="absolute top-4 left-6 flex items-center gap-2">
        <Activity className="w-3 h-3 text-studio-neon" />
        <span className="text-[10px] font-mono uppercase tracking-[0.2em] text-slate-400">Live Pitch Timeline</span>
      </div>
      <canvas ref={canvasRef} width={1200} height={300} className="w-full h-full" />
    </div>
  );
};

export default function App() {
  const [view, setView] = useState<View>(() => (localStorage.getItem('maestro_view') as View) || 'auth');
  const [track, setTrack] = useState<Track | null>(() => (localStorage.getItem('maestro_track') as Track) || null);
  const [vocalRange, setVocalRange] = useState<VocalRange | null>(() => {
    const saved = localStorage.getItem('maestro_range');
    return saved ? JSON.parse(saved) : null;
  });
  const [results, setResults] = useState<PitchResult[]>(() => {
    const saved = localStorage.getItem('maestro_results');
    return saved ? JSON.parse(saved) : [];
  });
  const [currentIndex, setCurrentIndex] = useState(-1);
  const [isRecording, setIsRecording] = useState(false);
  const [currentPitch, setCurrentPitch] = useState<number | null>(null);
  const [confidence, setConfidence] = useState(0);
  const [diagnosis, setDiagnosis] = useState<string | null>(() => localStorage.getItem('maestro_diagnosis'));
  const [isLoading, setIsLoading] = useState(false);
  const [calibrationStep, setCalibrationStep] = useState<'low' | 'high' | 'done'>('low');
  const [amplitude, setAmplitude] = useState(0);
  const [isLiveActive, setIsLiveActive] = useState(false);
  const [liveTranscript, setLiveTranscript] = useState<{ role: 'user' | 'maestro', text: string }[]>([]);

  const audioCtx = useRef<AudioContext | null>(null);
  const analyser = useRef<AnalyserNode | null>(null);
  const micStream = useRef<MediaStream | null>(null);
  const animationFrame = useRef<number | null>(null);
  const filterNode = useRef<BiquadFilterNode | null>(null);
  const liveSession = useRef<any>(null);
  const audioWorklet = useRef<ScriptProcessorNode | null>(null);
  const pitchHistory = useRef<number[]>([]);
  const smoothedPitch = useRef<number | null>(null);

  // Dynamic Notes based on range
  const getDynamicNotes = useCallback(() => {
    if (!vocalRange || track === 'Guitar') return track === 'Singing' ? SINGING_NOTES : GUITAR_NOTES;
    const baseFreq = vocalRange.low;
    const intervals = [0, 2, 4, 5, 7, 9, 11, 12];
    const names = ['Sa', 'Re', 'Ga', 'Ma', 'Pa', 'Da', 'Ni', "Sa'"];
    return intervals.map((step, i) => ({
      name: names[i],
      note: `Custom-${i}`,
      freq: baseFreq * Math.pow(2, step / 12)
    }));
  }, [vocalRange, track]);

  const notes = getDynamicNotes();

  // Persistence
  useEffect(() => {
    localStorage.setItem('maestro_view', view);
    if (track) localStorage.setItem('maestro_track', track);
    if (vocalRange) localStorage.setItem('maestro_range', JSON.stringify(vocalRange));
    if (results.length > 0) localStorage.setItem('maestro_results', JSON.stringify(results));
    if (diagnosis) localStorage.setItem('maestro_diagnosis', diagnosis);
  }, [view, track, vocalRange, results, diagnosis]);

  // --- Audio Logic ---

  const setupAudio = async () => {
    if (!audioCtx.current) {
      audioCtx.current = new (window.AudioContext || (window as any).webkitAudioContext)();
    }
    if (audioCtx.current.state === 'suspended') {
      await audioCtx.current.resume();
    }
    if (!micStream.current) {
      micStream.current = await navigator.mediaDevices.getUserMedia({ audio: true });
    }
    const source = audioCtx.current.createMediaStreamSource(micStream.current);
    if (!filterNode.current) {
      filterNode.current = audioCtx.current.createBiquadFilter();
      filterNode.current.type = 'lowpass';
      filterNode.current.frequency.setValueAtTime(1000, audioCtx.current.currentTime);
    }
    analyser.current = audioCtx.current.createAnalyser();
    analyser.current.fftSize = 4096; // Increased for better resolution
    source.connect(filterNode.current);
    filterNode.current.connect(analyser.current);
  };

  const updatePitch = useCallback(() => {
    if (!analyser.current || !audioCtx.current) return;
    const buffer = new Float32Array(analyser.current.fftSize);
    analyser.current.getFloatTimeDomainData(buffer);
    
    let sum = 0;
    for (let i = 0; i < buffer.length; i++) sum += buffer[i] * buffer[i];
    setAmplitude(Math.sqrt(sum / buffer.length));

    const { pitch, confidence: conf } = autoCorrelate(buffer, audioCtx.current.sampleRate);
    setConfidence(conf);
    
    if (pitch !== -1 && conf > 0.3) {
      pitchHistory.current.push(pitch);
      if (pitchHistory.current.length > 10) pitchHistory.current.shift();
      
      // Median filter for smoothing
      const sorted = [...pitchHistory.current].sort((a, b) => a - b);
      const median = sorted[Math.floor(sorted.length / 2)];
      smoothedPitch.current = median;
      setCurrentPitch(median);
    } else {
      pitchHistory.current = [];
      smoothedPitch.current = null;
      setCurrentPitch(null);
    }
    
    animationFrame.current = requestAnimationFrame(updatePitch);
  }, []);

  const playReference = async (freq: number) => {
    if (!freq || isNaN(freq)) return;
    try {
      await setupAudio();
      if (!audioCtx.current) return;
      
      const now = audioCtx.current.currentTime;
      
      // Create a vocal-like sound using a sawtooth wave and formant filters
      const osc = audioCtx.current.createOscillator();
      const gain = audioCtx.current.createGain();
      
      osc.type = 'sawtooth';
      osc.frequency.setValueAtTime(freq, now);
      
      // Formant filters for an "Ah" sound (approximate)
      const createFormant = (f: number, q: number, g: number) => {
        const filter = audioCtx.current!.createBiquadFilter();
        filter.type = 'bandpass';
        filter.frequency.value = f;
        filter.Q.value = q;
        const filterGain = audioCtx.current!.createGain();
        filterGain.gain.value = g;
        return { filter, filterGain };
      };

      const formant1 = createFormant(700, 12, 1.2);
      const formant2 = createFormant(1200, 12, 0.6);
      const formant3 = createFormant(2600, 12, 0.3);

      osc.connect(formant1.filter);
      osc.connect(formant2.filter);
      osc.connect(formant3.filter);

      formant1.filter.connect(formant1.filterGain);
      formant2.filter.connect(formant2.filterGain);
      formant3.filter.connect(formant3.filterGain);

      formant1.filterGain.connect(gain);
      formant2.filterGain.connect(gain);
      formant3.filterGain.connect(gain);

      gain.connect(audioCtx.current.destination);

      // Human-like Vibrato
      const vibrato = audioCtx.current.createOscillator();
      const vibratoGain = audioCtx.current.createGain();
      vibrato.frequency.value = 5.5; // 5.5Hz vibrato
      vibratoGain.gain.value = freq * 0.015; // 1.5% pitch depth
      vibrato.connect(vibratoGain);
      vibratoGain.connect(osc.frequency);
      vibrato.start(now);

      // Envelope
      gain.gain.setValueAtTime(0, now);
      gain.gain.linearRampToValueAtTime(0.15, now + 0.15);
      gain.gain.linearRampToValueAtTime(0.1, now + 0.8);
      gain.gain.linearRampToValueAtTime(0, now + 1.5);

      osc.start(now);
      osc.stop(now + 1.5);
      vibrato.stop(now + 1.5);
    } catch (err) {
      console.error("Audio error:", err);
    }
  };

  const singWithMaestro = async (noteName: string, freq: number) => {
    if (isLoading) return;
    setIsLoading(true);
    try {
      const response = await ai.models.generateContent({
        model: "gemini-2.5-flash-preview-tts",
        contents: [{ parts: [{ text: `Say cheerfully: "The note is ${noteName}". Then sing a long, steady "Ahhhh" at exactly ${freq} Hz. Make it sound like a professional vocal coach.` }] }],
        config: {
          responseModalities: [Modality.AUDIO],
          speechConfig: {
            voiceConfig: {
              prebuiltVoiceConfig: { voiceName: 'Kore' },
            },
          },
        },
      });

      const base64Audio = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
      if (base64Audio) {
        const audioData = atob(base64Audio);
        const arrayBuffer = new ArrayBuffer(audioData.length);
        const view = new Uint8Array(arrayBuffer);
        for (let i = 0; i < audioData.length; i++) {
          view[i] = audioData.charCodeAt(i);
        }
        
        await setupAudio();
        if (!audioCtx.current) return;
        
        const audioBuffer = audioCtx.current.createBuffer(1, view.length / 2, 24000);
        const floatView = audioBuffer.getChannelData(0);
        const int16View = new Int16Array(arrayBuffer);
        for (let i = 0; i < int16View.length; i++) {
          floatView[i] = int16View[i] / 32768;
        }
        
        const source = audioCtx.current.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(audioCtx.current.destination);
        source.start();
      }
    } catch (err) {
      console.error("Maestro singing error:", err);
      playReference(freq); // Fallback to synth
    } finally {
      setIsLoading(false);
    }
  };

  const startCalibration = async () => {
    try {
      await setupAudio();
      setIsRecording(true);
      setCalibrationStep('low');
      updatePitch();
    } catch (err) {
      alert("Please allow microphone access for calibration.");
    }
  };

  const captureCalibration = () => {
    if (!currentPitch) return;
    if (calibrationStep === 'low') {
      setVocalRange({ low: currentPitch, high: currentPitch });
      setCalibrationStep('high');
    } else if (calibrationStep === 'high') {
      setVocalRange(prev => prev ? { ...prev, high: currentPitch } : { low: currentPitch, high: currentPitch });
      setCalibrationStep('done');
      setIsRecording(false);
      if (animationFrame.current) cancelAnimationFrame(animationFrame.current);
      setView('test_select');
    }
  };

  const startTest = async (testType: View) => {
    try {
      await setupAudio();
      let initialResults: PitchResult[] = [];
      if (testType === 'pitch_test') {
        initialResults = notes.map(n => ({
          note: n.name, targetFreq: n.freq, detectedFreq: null, accuracy: 0, status: 'none'
        }));
      } else if (testType === 'interval_test') {
        const base = vocalRange?.low || 261.63;
        initialResults = [
          { note: 'Root', targetFreq: base, detectedFreq: null, accuracy: 0, status: 'none' },
          { note: 'Major 3rd', targetFreq: base * 1.25, detectedFreq: null, accuracy: 0, status: 'none' },
          { note: 'Perfect 5th', targetFreq: base * 1.5, detectedFreq: null, accuracy: 0, status: 'none' },
        ];
      } else if (testType === 'stability_test') {
        const mid = vocalRange ? (vocalRange.low + vocalRange.high) / 2 : 330;
        initialResults = [{ note: 'Hold Note', targetFreq: mid, detectedFreq: null, accuracy: 0, status: 'none' }];
      }
      setResults(initialResults);
      setCurrentIndex(0);
      setIsRecording(true);
      setView(testType);
      updatePitch();
    } catch (err) {
      alert("Please allow microphone access to take the test.");
    }
  };

  const captureNote = () => {
    if (currentIndex === -1 || !smoothedPitch.current) return;
    
    // Use the smoothed pitch for better accuracy
    const pitchToCapture = smoothedPitch.current;
    const target = results[currentIndex].targetFreq;
    const semitoneDiff = 12 * Math.log2(pitchToCapture / target);
    const accuracy = Math.max(0, 1 - Math.abs(semitoneDiff)); 
    let status: 'green' | 'yellow' | 'red' = 'red';
    if (accuracy > 0.94) status = 'green'; // Slightly stricter but more accurate
    else if (accuracy > 0.80) status = 'yellow';

    setResults(prev => {
      const next = [...prev];
      next[currentIndex] = { ...next[currentIndex], detectedFreq: pitchToCapture, accuracy, status };
      return next;
    });

    if (currentIndex < results.length - 1) setCurrentIndex(currentIndex + 1);
    else finishTest();
  };

  const finishTest = async () => {
    setIsRecording(false);
    if (animationFrame.current) cancelAnimationFrame(animationFrame.current);
    setIsLoading(true);
    setView('report');
    try {
      const response = await ai.models.generateContent({
        model: "gemini-3.1-pro-preview",
        config: {
          systemInstruction: "You are a world-class music coach and conservatory professor. Your goal is to provide highly accurate, technical, yet encouraging feedback on a student's pitch and stability performance. Analyze the raw data provided (target vs detected frequencies) and give a diagnosis that sounds professional and insightful. Use specific musical terminology (e.g., intonation, vibrato, timbre, cents, semitones).",
          thinkingConfig: { thinkingLevel: ThinkingLevel.HIGH },
        },
        contents: [{ role: 'user', parts: [{ text: `Analyze this performance data: ${JSON.stringify(results.map(r => ({ note: r.note, target: r.targetFreq.toFixed(2) + 'Hz', detected: r.detectedFreq ? r.detectedFreq.toFixed(2) + 'Hz' : 'N/A', accuracy: (r.accuracy * 100).toFixed(1) + '%', status: r.status })))}. Track: ${track}. Test: ${view}. Provide a detailed diagnosis and 2-3 specific technical drills to improve.` }] }],
      });
      setDiagnosis(response.text || "Report unavailable.");
    } catch (err) {
      setDiagnosis("Analysis failed.");
    } finally {
      setIsLoading(false);
    }
  };

  const stopLiveSession = useCallback(() => {
    setIsLiveActive(false);
    if (liveSession.current) {
      liveSession.current.close();
      liveSession.current = null;
    }
    if (audioWorklet.current) {
      audioWorklet.current.disconnect();
      audioWorklet.current = null;
    }
    setView('track_select');
  }, []);

  const startLiveSession = async () => {
    try {
      await setupAudio();
      if (!audioCtx.current || !micStream.current) return;
      
      setIsLiveActive(true);
      setLiveTranscript([]);
      setView('live_session');

      const session = await ai.live.connect({
        model: "gemini-3.1-flash-live-preview",
        config: {
          responseModalities: [Modality.AUDIO],
          systemInstruction: `You are Maestro, a world-class music coach and conservatory professor. 
          You are in a live session with a student. Listen to their singing or playing and provide real-time, encouraging, and technically accurate feedback. 
          If they are out of tune, gently guide them using musical terms (cents, semitones, intonation). 
          Keep your responses concise, musical, and professional. You can sing notes back to them if they ask.`,
          speechConfig: {
            voiceConfig: { prebuiltVoiceConfig: { voiceName: "Zephyr" } },
          },
        },
        callbacks: {
          onopen: () => {
            console.log("Live session opened");
            const processor = audioCtx.current!.createScriptProcessor(4096, 1, 1);
            audioWorklet.current = processor;
            
            const source = audioCtx.current!.createMediaStreamSource(micStream.current!);
            source.connect(processor);
            processor.connect(audioCtx.current!.destination);

            processor.onaudioprocess = (e) => {
              const inputData = e.inputBuffer.getChannelData(0);
              // Downsample to 16kHz PCM
              const pcmData = downsampleBuffer(inputData, audioCtx.current!.sampleRate, 16000);
              const base64Data = b64Encode(pcmData);
              session.sendRealtimeInput({
                audio: { data: base64Data, mimeType: 'audio/pcm;rate=16000' }
              });
            };
          },
          onmessage: async (message: LiveServerMessage) => {
            if (message.serverContent?.modelTurn) {
              const parts = message.serverContent.modelTurn.parts;
              for (const part of parts) {
                if (part.inlineData?.data) {
                  playLiveAudio(part.inlineData.data);
                }
                if (part.text) {
                  setLiveTranscript(prev => [...prev, { role: 'maestro', text: part.text! }]);
                }
              }
            }
            if (message.serverContent?.interrupted) {
              // Handle interruption if needed
            }
          },
          onclose: () => stopLiveSession(),
          onerror: (err) => {
            console.error("Live session error:", err);
            stopLiveSession();
          }
        }
      });
      liveSession.current = session;
    } catch (err) {
      console.error("Failed to start live session:", err);
      setIsLiveActive(false);
    }
  };

  const downsampleBuffer = (buffer: Float32Array, sampleRate: number, outSampleRate: number) => {
    if (outSampleRate === sampleRate) return new Int16Array(buffer.map(v => v * 0x7FFF));
    const sampleRateRatio = sampleRate / outSampleRate;
    const newLength = Math.round(buffer.length / sampleRateRatio);
    const result = new Int16Array(newLength);
    let offsetResult = 0;
    let offsetBuffer = 0;
    while (offsetResult < result.length) {
      const nextOffsetBuffer = Math.round((offsetResult + 1) * sampleRateRatio);
      let accum = 0, count = 0;
      for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
        accum += buffer[i];
        count++;
      }
      result[offsetResult] = Math.min(1, Math.max(-1, accum / count)) * 0x7FFF;
      offsetResult++;
      offsetBuffer = nextOffsetBuffer;
    }
    return result;
  };

  const b64Encode = (buffer: Int16Array) => {
    const bytes = new Uint8Array(buffer.buffer);
    let binary = '';
    for (let i = 0; i < bytes.byteLength; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    return btoa(binary);
  };

  const playLiveAudio = (base64Data: string) => {
    if (!audioCtx.current) return;
    const binary = atob(base64Data);
    const buffer = new ArrayBuffer(binary.length);
    const bytes = new Uint8Array(buffer);
    for (let i = 0; i < binary.length; i++) {
      bytes[i] = binary.charCodeAt(i);
    }
    const int16Data = new Int16Array(buffer);
    const floatData = new Float32Array(int16Data.length);
    for (let i = 0; i < int16Data.length; i++) {
      floatData[i] = int16Data[i] / 32768;
    }
    
    const audioBuffer = audioCtx.current.createBuffer(1, floatData.length, 24000);
    audioBuffer.getChannelData(0).set(floatData);
    const source = audioCtx.current.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(audioCtx.current.destination);
    source.start();
  };

  const reset = () => {
    localStorage.clear();
    window.location.reload();
  };

  // --- Views ---

  if (view === 'auth') return (
    <div className="min-h-screen flex flex-col items-center justify-center p-6 text-center space-y-12">
      <VibeBackground amplitude={0.05} />
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="space-y-4">
        <div className="inline-flex items-center gap-3 bg-studio-accent/5 px-4 py-2 rounded-full border border-studio-accent/10">
          <Sparkles className="w-4 h-4 text-studio-neon" />
          <span className="text-xs font-bold uppercase tracking-[0.2em] text-studio-neon">Maestro AI v2.0</span>
        </div>
        <h1 className="text-7xl font-black text-slate-900 tracking-tighter leading-none font-serif">
          Master Your <br />
          <span className="text-studio-neon italic">Instrument.</span>
        </h1>
        <p className="text-slate-600 text-xl max-w-md mx-auto font-medium">
          The world's first AI-powered studio coach for singers and guitarists.
        </p>
      </motion.div>
      <div className="grid grid-cols-1 gap-4 w-full max-w-xs">
        <button onClick={() => setView('track_select')} className="studio-button-primary flex items-center justify-center gap-3 text-lg">
          <Chrome className="w-5 h-5" /> Continue with Google
        </button>
        <button onClick={() => setView('track_select')} className="studio-button-secondary flex items-center justify-center gap-3">
          <Phone className="w-5 h-5" /> Phone Number
        </button>
      </div>
    </div>
  );

  if (view === 'track_select') return (
    <div className="min-h-screen flex flex-col items-center justify-center p-6 space-y-12">
      <VibeBackground amplitude={0.02} />
      <div className="text-center space-y-2">
        <h2 className="text-sm font-bold uppercase tracking-[0.3em] text-studio-neon">Step 01</h2>
        <h1 className="text-5xl font-black text-slate-900 tracking-tight font-serif">Choose Your Path</h1>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 w-full max-w-3xl">
        <TrackCard title="Singing" icon={<Mic className="w-12 h-12" />} desc="Calibrate your voice and master pitch, intervals, and stability." onClick={() => { setTrack('Singing'); setView('calibration'); }} />
        <TrackCard title="Guitar" icon={<Guitar className="w-12 h-12" />} desc="Perfect your tuning and master complex chromatic runs." onClick={() => { setTrack('Guitar'); setView('test_select'); }} />
      </div>
      <motion.button 
        whileHover={{ scale: 1.05 }} 
        whileTap={{ scale: 0.95 }}
        onClick={startLiveSession}
        className="studio-button-primary flex items-center gap-3 px-10 py-5 text-xl shadow-2xl shadow-amber-900/20"
      >
        <Sparkles className="w-6 h-6" /> Live Maestro Session
      </motion.button>
    </div>
  );

  if (view === 'live_session') return (
    <div className="min-h-screen flex flex-col p-6 space-y-8">
      <VibeBackground amplitude={amplitude} />
      <header className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <button onClick={stopLiveSession} className="studio-button-secondary p-3 rounded-full"><RotateCcw className="w-5 h-5" /></button>
          <div>
            <h2 className="text-xs font-bold uppercase tracking-widest text-studio-neon">Live Session</h2>
            <h1 className="text-2xl font-black text-slate-900 font-serif">Maestro is Listening</h1>
          </div>
        </div>
        <div className="flex items-center gap-3 bg-green-500/10 text-green-600 px-4 py-2 rounded-full border border-green-500/20">
          <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
          <span className="text-xs font-bold uppercase tracking-widest">Live</span>
        </div>
      </header>

      <div className="flex-1 grid grid-cols-1 lg:grid-cols-3 gap-8 overflow-hidden">
        <div className="lg:col-span-2 flex flex-col space-y-6">
          <div className="glass-card flex-1 p-8 flex flex-col items-center justify-center space-y-12 relative overflow-hidden">
            <motion.div 
              animate={{ 
                scale: 1 + amplitude * 5,
                opacity: 0.2 + amplitude * 0.8
              }}
              className="w-64 h-64 bg-studio-accent/20 rounded-full blur-3xl"
            />
            <div className="absolute inset-0 flex flex-col items-center justify-center space-y-6">
              <div className="w-32 h-32 rounded-full bg-studio-accent flex items-center justify-center shadow-2xl shadow-amber-900/40">
                <Mic className="w-16 h-16 text-white" />
              </div>
              <div className="text-center space-y-2">
                <div className="text-4xl font-black text-slate-900 font-serif italic">
                  {currentPitch ? `${currentPitch.toFixed(1)} Hz` : '...'}
                </div>
                <p className="text-studio-neon font-bold uppercase tracking-widest text-[10px]">Real-time Pitch Tracking</p>
              </div>
            </div>
          </div>
          
          <div className="glass-card p-6 flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="w-10 h-10 rounded-full bg-studio-accent/10 flex items-center justify-center text-studio-neon">
                <Volume2 className="w-5 h-5" />
              </div>
              <div>
                <h4 className="font-bold text-slate-900 text-sm">Maestro's Voice</h4>
                <p className="text-[10px] text-slate-500">Active & Ready to guide you</p>
              </div>
            </div>
            <button onClick={stopLiveSession} className="studio-button-primary !bg-red-500 hover:!bg-red-600 !shadow-red-900/20">End Session</button>
          </div>
        </div>

        <div className="glass-card flex flex-col overflow-hidden">
          <div className="p-4 border-b border-studio-border flex items-center gap-2">
            <MessageSquare className="w-4 h-4 text-studio-neon" />
            <h3 className="font-bold text-slate-900 text-sm font-serif">Session Transcript</h3>
          </div>
          <div className="flex-1 overflow-y-auto p-6 space-y-6">
            {liveTranscript.length === 0 ? (
              <div className="h-full flex flex-col items-center justify-center text-center space-y-4 opacity-40">
                <Sparkles className="w-8 h-8 text-studio-neon" />
                <p className="text-xs font-medium max-w-[150px]">Start singing or ask Maestro a question to begin.</p>
              </div>
            ) : (
              liveTranscript.map((msg, i) => (
                <motion.div 
                  initial={{ opacity: 0, x: -10 }} 
                  animate={{ opacity: 1, x: 0 }} 
                  key={i} 
                  className="space-y-1"
                >
                  <span className="text-[10px] font-bold uppercase tracking-widest text-studio-neon">{msg.role}</span>
                  <p className="text-sm text-slate-700 leading-relaxed">{msg.text}</p>
                </motion.div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );

  if (view === 'calibration') return (
    <div className="min-h-screen flex flex-col items-center justify-center p-6 text-center space-y-8">
      <VibeBackground amplitude={isRecording ? amplitude : 0.02} />
      <div className="space-y-2">
        <h2 className="text-sm font-bold uppercase tracking-[0.3em] text-studio-neon">Calibration</h2>
        <h1 className="text-5xl font-black text-slate-900 tracking-tight font-serif">Personalize Maestro</h1>
      </div>
      <div className="glass-card p-12 w-full max-w-md space-y-10 relative overflow-hidden">
        <div className="absolute top-0 left-0 w-full h-1 bg-studio-border">
          <motion.div className="h-full bg-studio-neon shadow-[0_0_10px_rgba(197,160,89,0.5)]" animate={{ width: calibrationStep === 'low' ? '33%' : calibrationStep === 'high' ? '66%' : '100%' }} />
        </div>
        <div className="space-y-4">
          <div className="text-6xl font-black text-slate-900 font-serif italic">{calibrationStep === 'low' ? 'Low Note' : calibrationStep === 'high' ? 'High Note' : 'Done!'}</div>
          <p className="text-studio-neon font-bold uppercase tracking-widest text-xs">{calibrationStep === 'low' ? 'Sing your lowest comfortable note' : 'Sing your highest comfortable note'}</p>
        </div>
        {!isRecording ? (
          <button onClick={startCalibration} className="studio-button-primary w-full py-5 text-lg">Start Calibration</button>
        ) : (
          <div className="space-y-8">
            <div className="text-3xl font-mono font-black text-slate-900">{currentPitch ? `${currentPitch.toFixed(1)} Hz` : 'Listening...'}</div>
            <button onClick={captureCalibration} className="studio-button-primary w-full py-5 text-lg">Capture {calibrationStep === 'low' ? 'Low' : 'High'}</button>
          </div>
        )}
      </div>
    </div>
  );

  if (view === 'test_select') return (
    <div className="min-h-screen flex flex-col items-center justify-center p-6 space-y-12">
      <VibeBackground amplitude={0.02} />
      <div className="text-center space-y-2">
        <h2 className="text-sm font-bold uppercase tracking-[0.3em] text-studio-neon">Dashboard</h2>
        <h1 className="text-5xl font-black text-slate-900 tracking-tight font-serif">Choose Your Test</h1>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 w-full max-w-5xl">
        <TestCard title="Basic Pitch" desc="Sing/play the standard scale." icon={<Music className="w-8 h-8" />} onClick={() => startTest('pitch_test')} />
        <TestCard title="Interval Jumps" desc="Test your relative pitch accuracy." icon={<Guitar className="w-8 h-8" />} onClick={() => startTest('interval_test')} />
        <TestCard title="Stability" desc="Hold a note perfectly steady." icon={<Mic className="w-8 h-8" />} onClick={() => startTest('stability_test')} />
      </div>
      <div className="glass-card p-6 w-full max-w-5xl flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 rounded-full bg-studio-accent/10 flex items-center justify-center text-studio-neon"><Trophy className="w-6 h-6" /></div>
          <div><h4 className="font-bold text-slate-900 font-serif">Daily Goal</h4><p className="text-xs text-slate-500">Complete 3 tests today to keep your streak.</p></div>
        </div>
        <div className="flex gap-2">{[1, 2, 3].map(i => <div key={i} className="w-3 h-3 rounded-full bg-studio-border" />)}</div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen flex flex-col font-sans text-slate-600">
      <VibeBackground amplitude={isRecording ? amplitude : 0.01} />
      <header className="glass-card !rounded-none border-t-0 border-x-0 p-4 flex items-center justify-between sticky top-0 z-10">
        <button 
          onClick={() => setView('auth')}
          className="flex items-center gap-2 hover:opacity-80 transition-opacity active:scale-95"
        >
          <div className="bg-studio-accent p-1.5 rounded-lg">
            <Sparkles className="w-4 h-4 text-white" />
          </div>
          <span className="font-black text-slate-900 tracking-tight font-serif text-xl">Maestro AI</span>
        </button>
        <div className="flex items-center gap-4">
          <div className="text-[10px] font-mono uppercase tracking-widest text-amber-900/40 bg-amber-50 px-2 py-1 rounded">{track} Mode</div>
          <button onClick={reset} className="text-slate-400 hover:text-slate-900 transition-colors"><LogOut className="w-5 h-5" /></button>
        </div>
      </header>
      <main className="flex-1 p-6 md:p-12 max-w-5xl mx-auto w-full">
        {view === 'report' ? (
          <div className="space-y-12">
            <div className="flex items-center justify-between">
              <div className="space-y-1">
                <h2 className="text-sm font-bold uppercase tracking-[0.3em] text-studio-neon">Analysis</h2>
                <h1 className="text-5xl font-black text-slate-900 tracking-tight font-serif">Performance Report</h1>
              </div>
              <button onClick={() => setView('test_select')} className="studio-button-secondary flex items-center gap-2"><RotateCcw className="w-4 h-4" /> New Test</button>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {results.map((r, i) => (
                <div key={i} className="glass-card p-6 flex flex-col items-center gap-3">
                  <span className="text-[10px] font-mono uppercase tracking-widest text-slate-400">{r.note}</span>
                  <div className={`w-12 h-12 rounded-full flex items-center justify-center ${r.status === 'green' ? 'bg-green-500/10 text-green-600' : r.status === 'yellow' ? 'bg-amber-500/10 text-amber-600' : 'bg-red-500/10 text-red-600'}`}>
                    {r.status === 'green' ? <CheckCircle2 className="w-6 h-6" /> : r.status === 'yellow' ? <AlertCircle className="w-6 h-6" /> : <XCircle className="w-6 h-6" />}
                  </div>
                  <span className="text-xl font-black text-slate-900 font-serif">{(r.accuracy * 100).toFixed(0)}%</span>
                </div>
              ))}
            </div>
            <div className="glass-card p-10 space-y-8">
              <div className="flex items-center gap-3 text-studio-neon font-black text-2xl font-serif italic"><Sparkles className="w-6 h-6" /> <h2>Maestro's Analysis</h2></div>
              {isLoading ? (
                <div className="space-y-6">
                  <div className="flex items-center gap-4"><div className="w-10 h-10 bg-studio-accent/10 rounded-full flex items-center justify-center animate-bounce"><Zap className="w-5 h-5 text-studio-neon" /></div><p className="text-slate-500 font-medium animate-pulse">Maestro is decoding your performance...</p></div>
                  <div className="space-y-3 pl-14"><div className="h-2 bg-amber-50 rounded-full w-3/4 animate-pulse" /><div className="h-2 bg-amber-50 rounded-full w-full animate-pulse" /><div className="h-2 bg-amber-50 rounded-full w-2/3 animate-pulse" /></div>
                </div>
              ) : (
                <div className="markdown-body"><Markdown>{diagnosis || ""}</Markdown></div>
              )}
            </div>
          </div>
        ) : (
          <div className="space-y-12">
            <div className="text-center space-y-2">
              <h2 className="text-sm font-bold uppercase tracking-[0.3em] text-studio-neon">{view === 'pitch_test' ? 'Test 01' : view === 'interval_test' ? 'Test 02' : 'Test 03'}</h2>
              <h1 className="text-5xl font-black text-slate-900 tracking-tight font-serif">{view === 'pitch_test' ? 'Basic Pitch' : view === 'interval_test' ? 'Interval Jumps' : 'Stability Test'}</h1>
            </div>
            {!isRecording ? (
              <motion.button whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }} onClick={() => startTest(view)} className="studio-button-primary px-12 py-6 text-xl flex items-center gap-3 mx-auto"><Play className="w-6 h-6 fill-current" /> Begin Session</motion.button>
            ) : (
              <div className="space-y-12">
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 items-start">
                  <div className="glass-card p-12 flex flex-col items-center justify-center space-y-6 text-center lg:col-span-1">
                    <AnimatePresence mode="wait"><motion.div key={currentIndex} initial={{ opacity: 0, scale: 0.5 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 1.5 }} className="text-[12rem] font-black text-slate-900 leading-none tracking-tighter font-serif italic">{results[currentIndex]?.note}</motion.div></AnimatePresence>
                    <div className="space-y-1"><div className="text-[10px] font-mono uppercase tracking-widest text-slate-400">Target Frequency</div><div className="text-xl font-black text-studio-neon">{results[currentIndex]?.targetFreq.toFixed(1)} Hz</div></div>
                    <div className="w-full h-1 bg-studio-border rounded-full overflow-hidden mt-4">
                      <motion.div 
                        animate={{ width: `${confidence * 100}%`, backgroundColor: confidence > 0.5 ? '#C5A059' : '#ef4444' }}
                        className="h-full transition-colors"
                      />
                    </div>
                    <div className="text-[8px] font-bold uppercase tracking-widest text-slate-400 mt-1">Signal Confidence</div>
                  </div>
                  <div className="lg:col-span-2 space-y-8">
                    <div className="relative">
                      <PitchGraph currentPitch={currentPitch} targetFreq={results[currentIndex]?.targetFreq} isRecording={isRecording} />
                      {currentPitch && (
                        <motion.div 
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          className="absolute top-4 right-4 glass-card px-4 py-2 flex flex-col items-end"
                        >
                          <span className="text-[8px] font-bold uppercase tracking-widest text-slate-400">Pitch Offset</span>
                          <span className={`text-xl font-mono font-black ${
                            Math.abs(1200 * Math.log2(currentPitch / results[currentIndex].targetFreq)) < 20 
                              ? 'text-green-600' 
                              : Math.abs(1200 * Math.log2(currentPitch / results[currentIndex].targetFreq)) < 50 
                                ? 'text-amber-600' 
                                : 'text-red-600'
                          }`}>
                            {Math.round(1200 * Math.log2(currentPitch / results[currentIndex].targetFreq))} cents
                          </span>
                        </motion.div>
                      )}
                    </div>
                    <div className="flex flex-wrap justify-center gap-4">
                      <button 
                        onClick={() => results[currentIndex] && playReference(results[currentIndex].targetFreq)} 
                        className="studio-button-secondary flex items-center gap-2"
                      >
                        <Music className="w-5 h-5" /> Sing Note
                      </button>
                      <button 
                        onClick={() => results[currentIndex] && singWithMaestro(results[currentIndex].note, results[currentIndex].targetFreq)} 
                        disabled={isLoading}
                        className="studio-button-secondary flex items-center gap-2 border-studio-neon/30 hover:border-studio-neon"
                      >
                        <Sparkles className={`w-5 h-5 text-studio-neon ${isLoading ? 'animate-spin' : ''}`} /> 
                        Maestro Sing
                      </button>
                      <button onClick={captureNote} className="studio-button-primary flex items-center gap-2 px-12">Capture Note</button>
                    </div>
                    <div className="grid grid-cols-4 md:grid-cols-8 gap-3">
                      {results.map((r, i) => (
                        <div key={i} className={`h-14 rounded-2xl flex items-center justify-center font-black text-lg transition-all border-2 ${i === currentIndex ? 'border-studio-neon bg-studio-accent/5 text-studio-neon shadow-[0_0_15px_rgba(197,160,89,0.1)]' : r.status === 'green' ? 'bg-green-500/10 border-green-500/30 text-green-600' : r.status === 'yellow' ? 'bg-amber-500/10 border-amber-500/30 text-amber-600' : r.status === 'red' ? 'bg-red-500/10 border-red-500/30 text-red-600' : 'bg-white border-slate-100 text-slate-300'}`}>{r.note}</div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}

// --- Sub-Components ---

const TrackCard = ({ title, icon, desc, onClick }: { title: string, icon: React.ReactNode, desc: string, onClick: () => void }) => (
  <motion.button whileHover={{ y: -8, scale: 1.02 }} whileTap={{ scale: 0.98 }} onClick={onClick} className="glass-card p-10 text-left space-y-6 group relative overflow-hidden">
    <div className="absolute top-0 right-0 w-32 h-32 bg-studio-accent/5 rounded-full -mr-16 -mt-16 blur-3xl group-hover:bg-studio-accent/10 transition-colors" />
    <div className="bg-studio-accent/5 w-20 h-20 rounded-2xl flex items-center justify-center text-studio-neon group-hover:bg-studio-accent group-hover:text-white transition-all duration-500">{icon}</div>
    <div className="space-y-2"><h3 className="text-3xl font-black text-slate-900 font-serif">{title}</h3><p className="text-slate-600 font-medium leading-relaxed">{desc}</p></div>
    <div className="flex items-center gap-2 text-studio-neon font-bold group-hover:gap-4 transition-all">Start Training <ArrowRight className="w-4 h-4" /></div>
  </motion.button>
);

const TestCard = ({ title, desc, icon, onClick }: { title: string, desc: string, icon: React.ReactNode, onClick: () => void }) => (
  <motion.button whileHover={{ y: -4 }} whileTap={{ scale: 0.98 }} onClick={onClick} className="glass-card p-8 text-left space-y-4 group">
    <div className="text-studio-neon group-hover:scale-110 transition-transform duration-500">{icon}</div>
    <div className="space-y-1"><h3 className="text-xl font-black text-slate-900 font-serif">{title}</h3><p className="text-slate-500 text-sm font-medium">{desc}</p></div>
  </motion.button>
);

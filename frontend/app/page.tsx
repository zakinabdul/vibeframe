"use client";

import { type CSSProperties, useEffect, useMemo, useRef, useState } from "react";
import WaveformVisualizer from "../components/WaveformVisualizer";
import styles from "./page.module.css";

type Critique = {
  score: number;
  issues: string[];
  suggestions: string[];
};

type GenerateResponse = {
  artboard_id: string | null;
  palette_artboard_id: string | null;
  node_ids: string[];
  html_used: string;
  round: number;
  critique: Critique;
  done: boolean;
  assistant_message: string;
  questions: string[];
  conversation_stage: string;
};

type StreamEvent = {
  type?: string;
  provider?: string;
  agent?: string;
  message?: string;
  questions?: string[];
  round?: number;
};

type AgentEvent = {
  id: string;
  agent: string;
  type: string;
  message: string;
  timestamp: number;
};

type ChatTurn = {
  id: string;
  role: "assistant" | "user" | "system";
  text: string;
  meta: string;
};

type PaperOpenResponse = {
  opened: boolean;
  message: string;
};

type TranscribeResponse = {
  text: string;
  provider: string;
  model: string;
};

type VisualState = "idle" | "listening" | "thinking" | "speaking";

function backendUrl() {
  return process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://127.0.0.1:8000";
}

function newConversationId() {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `conversation-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

const AGENT_LABELS: Record<string, string> = {
  brief_agent: "Brief Agent",
  design_agent: "Design Agent",
  critique_agent: "Critique Agent",
  orchestrator: "Orchestrator",
  voice_agent: "Voice Agent",
  system: "System",
  brief: "Brief Agent",
  design: "Design Agent",
  critique: "Critique Agent",
};

const AGENT_COLORS: Record<string, string> = {
  brief_agent: "#6f75ff",
  design_agent: "#1fd47f",
  critique_agent: "#f5a623",
  orchestrator: "#c46dff",
  voice_agent: "#38bdf8",
  system: "#6b7280",
  brief: "#6f75ff",
  design: "#1fd47f",
  critique: "#f5a623",
};

function hexToRgba(hex: string, alpha: number) {
  const r = parseInt(hex.slice(1, 3), 16) || 107;
  const g = parseInt(hex.slice(3, 5), 16) || 114;
  const b = parseInt(hex.slice(5, 7), 16) || 128;
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function parseAgentEvent(raw: string, idx: number): AgentEvent {
  try {
    const parsed = JSON.parse(raw) as StreamEvent;
    
    let agentKey = parsed.agent ?? "system";
    if (parsed.provider) {
        agentKey = parsed.provider.toLowerCase().replace("agent", "").replace("_", "");
    }
    if (parsed.type && agentKey === "system") {
        const typeL = parsed.type.toLowerCase();
        if (typeL.includes('design')) agentKey = 'design_agent';
        else if (typeL.includes('critic')) agentKey = 'critique_agent';
        else if (typeL.includes('brief')) agentKey = 'brief_agent';
        else if (typeL.includes('orchestrator')) agentKey = 'orchestrator';
    }

    const message =
      parsed.message ||
      (Array.isArray(parsed.questions) && parsed.questions.length
        ? `${parsed.questions.length} clarifying question${parsed.questions.length > 1 ? "s" : ""} generated`
        : parsed.type?.replaceAll("_", " ") || raw);
    return {
      id: `${idx}-${Date.now()}`,
      agent: agentKey,
      type: parsed.type ?? "event",
      message,
      timestamp: Date.now(),
    };
  } catch {
    return { id: `raw-${idx}`, agent: "system", type: "raw", message: raw, timestamp: Date.now() };
  }
}

function toErrorMessage(payload: unknown, fallback: string) {
  if (typeof payload === "object" && payload && "detail" in payload && typeof payload.detail === "string") {
    return payload.detail;
  }
  return fallback;
}

function getAgentLabel(agent: string) {
  if (agent.includes("critic")) return "Critic";
  if (agent.includes("design") || agent.includes("brief") || agent.includes("orchestrator")) return "Designer";
  return "System";
}

function getAgentColor(agent: string) {
  if (agent.includes("critic")) return "#f59e0b";
  if (agent.includes("design") || agent.includes("brief") || agent.includes("orchestrator")) return "#6366f1";
  return "#22c55e";
}

function formatTime(timestamp: number) {
  return new Date(timestamp).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });
}

export default function HomePage() {
  const api = useMemo(() => backendUrl(), []);
  const wsUrl = useMemo(() => {
    const base = backendUrl();
    return base.replace(/^http/, "ws") + "/ws/transcribe";
  }, []);
  const [conversationId, setConversationId] = useState(() => newConversationId());

  const [message, setMessage] = useState("");
  const [transcript, setTranscript] = useState("");
  const [stage, setStage] = useState("intake");
  const [questions, setQuestions] = useState<string[]>([]);
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const logsEndRef = useRef<HTMLDivElement>(null);
  const [chat, setChat] = useState<ChatTurn[]>([]);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const briefRef = useRef<HTMLTextAreaElement>(null);
  const [autoSend, setAutoSend] = useState(false);
  const [toast, setToast] = useState("");
  const [isSpeaking, setIsSpeaking] = useState(false);

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [chat]);
  const [busy, setBusy] = useState(false);
  const [listening, setListening] = useState(false);
  const [voiceSupported, setVoiceSupported] = useState(false);
  const [status, setStatus] = useState("Idle");
  const [error, setError] = useState("");
  const [round, setRound] = useState(0);

  const [logsMinimized, setLogsMinimized] = useState(false);
  const [toastVisible, setToastVisible] = useState(false);
  const [chatPaneSize, setChatPaneSize] = useState(58);
  const isResizingRef = useRef(false);

  const recognitionRef = useRef<SpeechRecognition | null>(null);
  const transcriptBufferRef = useRef("");
  const interimTranscriptRef = useRef("");
  const manualStopRef = useRef(false);
  const restartTimerRef = useRef<number | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const mediaChunksRef = useRef<Blob[]>([]);
  const pendingRecorderStopRef = useRef<Promise<Blob | null> | null>(null);
  const voiceModeActiveRef = useRef(false);
  const spacebarHeldRef = useRef(false);
  const autoSendRef = useRef(false);

  // WebSocket ref for streaming transcription
  const wsRef = useRef<WebSocket | null>(null);
  const wsConnectedRef = useRef(false);

  // Silence detection: auto-stop after 800ms of no speech
  const silenceTimerRef = useRef<number | null>(null);
  const SILENCE_TIMEOUT_MS = 800;
  const lastSpeechTimeRef = useRef(0);

  useEffect(() => {
    if (events.length > 0) {
      logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [events]);

  useEffect(() => {
    if (!toast) {
      setToastVisible(false);
      return;
    }

    setToastVisible(true);
    const timer = window.setTimeout(() => {
      setToastVisible(false);
      setToast("");
    }, 3000);

    return () => window.clearTimeout(timer);
  }, [toast]);

  useEffect(() => {
    const EventSourceImpl = window.EventSource;
    const source = new EventSourceImpl(`${api}/stream`);
    let idx = 0;

    source.onmessage = (event) => {
      if (!event.data) return;
      const agentEvent = parseAgentEvent(event.data, idx++);
      setEvents((current) => [...current, agentEvent].slice(-40));

      try {
        const parsed = JSON.parse(event.data) as StreamEvent;
        if (parsed.type) {
          setStatus(parsed.type.replaceAll("_", " "));
        }
      } catch {
        // Keep raw stream display even when not JSON.
      }
    };

    source.onerror = () => {
      const errEvent: AgentEvent = {
        id: `err-${Date.now()}`,
        agent: "system",
        type: "disconnected",
        message: "Stream disconnected",
        timestamp: Date.now(),
      };
      setEvents((current) => [...current, errEvent].slice(-40));
    };

    return () => source.close();
  }, [api]);

  useEffect(() => {
    const onMouseMove = (event: MouseEvent) => {
      if (!isResizingRef.current) return;
      const rightPanel = document.getElementById("right-panel");
      if (!rightPanel) return;
      const rect = rightPanel.getBoundingClientRect();
      const relativeY = event.clientY - rect.top;
      const ratio = (relativeY / Math.max(rect.height, 1)) * 100;
      const bounded = Math.min(78, Math.max(28, ratio));
      setChatPaneSize(bounded);
    };

    const onMouseUp = () => {
      isResizingRef.current = false;
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };

    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
    return () => {
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onMouseUp);
    };
  }, []);

  useEffect(() => {
    const SpeechRecognitionApi = window.SpeechRecognition ?? window.webkitSpeechRecognition;
    const mediaRecorderAvailable =
      typeof window.MediaRecorder !== "undefined" &&
      typeof navigator !== "undefined" &&
      Boolean(navigator.mediaDevices?.getUserMedia);
    setVoiceSupported(Boolean(SpeechRecognitionApi) || mediaRecorderAvailable);

    // Warm up the WebSocket connection immediately to avoid initial latency
    connectTranscribeWs();
  }, [wsUrl]);

  useEffect(() => {
    return () => {
      if (restartTimerRef.current !== null) {
        window.clearTimeout(restartTimerRef.current);
      }
      if (silenceTimerRef.current !== null) {
        window.clearTimeout(silenceTimerRef.current);
      }
      const recorder = mediaRecorderRef.current;
      if (recorder && recorder.state !== "inactive") {
        recorder.stop();
      }
      if (mediaStreamRef.current) {
        for (const track of mediaStreamRef.current.getTracks()) {
          track.stop();
        }
      }
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
    };
  }, []);

  useEffect(() => {
    autoSendRef.current = autoSend;
  }, [autoSend]);

  // Spacebar hold-to-record: press = start, release = stop + send
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName?.toLowerCase();
      // Don't intercept space inside textareas/inputs
      if (tag === "textarea" || tag === "input" || e.repeat) return;
      if (e.code === "Space" && !spacebarHeldRef.current && !busy) {
        e.preventDefault();
        spacebarHeldRef.current = true;
        startVoice();
      }
    };
    const onKeyUp = (e: KeyboardEvent) => {
      if (e.code === "Space" && spacebarHeldRef.current) {
        spacebarHeldRef.current = false;
        stopVoice();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [busy]);

  function pushChat(role: ChatTurn["role"], text: string, meta: string) {
    setChat((current) => [
      ...current,
      {
        id: `${role}-${Date.now()}-${Math.random().toString(16).slice(2)}`,
        role,
        text,
        meta,
      },
    ]);
  }

  async function resetSession() {
    voiceModeActiveRef.current = false;
    spacebarHeldRef.current = false;
    manualStopRef.current = true;
    if (restartTimerRef.current !== null) {
      window.clearTimeout(restartTimerRef.current);
      restartTimerRef.current = null;
    }
    if (silenceTimerRef.current !== null) {
      window.clearTimeout(silenceTimerRef.current);
      silenceTimerRef.current = null;
    }
    if (window.speechSynthesis) {
      window.speechSynthesis.cancel();
    }
    if (recognitionRef.current) {
      recognitionRef.current.stop();
    }
    await stopAudioCapture();
    // Reset WebSocket
    if (wsRef.current) {
      try { wsRef.current.send(JSON.stringify({ action: "reset" })); } catch { /* ignore */ }
    }
    // Tell backend to clear the session so the next run creates a fresh canvas
    try {
      await fetch(`${api}/session/reset`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ conversation_id: conversationId }),
      });
    } catch {
      // Ignore network errors on reset
    }
    setConversationId(newConversationId());
    setStage("intake");
    setQuestions([]);
    setRound(0);
    setError("");
    setTranscript("");
    setMessage("");
    setStatus("New session");
    setChat([]);
    setToast("");
  }

  function speak(text: string) {
    if (!text || !window.speechSynthesis) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "en-US";
    utterance.onstart = () => setIsSpeaking(true);
    utterance.onend = () => {
      setIsSpeaking(false);
      if (voiceModeActiveRef.current && !busy) {
        startVoice();
      }
    };
    utterance.onerror = () => setIsSpeaking(false);
    window.speechSynthesis.speak(utterance);
  }

  async function handleSend(text: string, source: "text" | "voice") {
    const trimmed = text.trim();
    if (trimmed.length < 3) {
      setError("Please enter at least 3 characters.");
      return;
    }

    if (source === "text") {
      voiceModeActiveRef.current = false;
    }
    if (window.speechSynthesis) {
      window.speechSynthesis.cancel();
    }

    setBusy(true);
    setError("");
    pushChat("user", trimmed, source === "voice" ? "You (voice)" : "You");
    // Clear the voice transcript buffer immediately after any send (text or voice)
    setTranscript("");
    transcriptBufferRef.current = "";
    interimTranscriptRef.current = "";
    setMessage("");

    try {
      const response = await fetch(`${api}/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          brief: trimmed,
          source,
          conversation_id: conversationId,
        }),
      });

      const raw = (await response.json()) as GenerateResponse | { detail?: string };
      if (!response.ok) {
        throw new Error(toErrorMessage(raw, "Generate request failed."));
      }

      const data = raw as GenerateResponse;
      setStage(data.conversation_stage || "building");
      setQuestions(data.questions || []);
      setRound(data.round || 0);
      const msg = data.assistant_message || "Working...";
      pushChat("assistant", msg, "AI Agent");
      speak(msg);
      // Note: questions are embedded in the AI's assistant_message — no separate system bubble needed
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Failed to call backend.";
      setError(msg);
      pushChat("system", msg, "System");
    } finally {
      setBusy(false);
    }
  }

  async function transcribeAudioBlob(blob: Blob): Promise<string> {
    const formData = new FormData();
    const fileName = blob.type.includes("wav") ? "voice.wav" : "voice.webm";
    formData.append("audio", blob, fileName);
    formData.append("language", navigator.language || "en-US");

    const response = await fetch(`${api}/transcribe`, {
      method: "POST",
      body: formData,
    });

    const payload = (await response.json()) as TranscribeResponse | { detail?: string };
    if (!response.ok) {
      throw new Error(toErrorMessage(payload, "Audio transcription failed."));
    }
    return (payload as TranscribeResponse).text?.trim() ?? "";
  }

  async function startAudioCapture(): Promise<boolean> {
    if (!navigator.mediaDevices?.getUserMedia || typeof window.MediaRecorder === "undefined") {
      return false;
    }

    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      return true;
    }

    try {
      let stream = mediaStreamRef.current;
      if (!stream) {
        stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
            channelCount: 1,
            sampleRate: 16000,
          },
        });
        mediaStreamRef.current = stream;
      }

      mediaChunksRef.current = [];

      const mimeCandidates = [
        "audio/webm;codecs=opus",
        "audio/webm",
        "audio/mp4",
      ];
      const supportedMime = mimeCandidates.find((candidate) => MediaRecorder.isTypeSupported(candidate));
      const recorder = supportedMime
        ? new MediaRecorder(stream, { mimeType: supportedMime })
        : new MediaRecorder(stream);

      recorder.ondataavailable = (event: BlobEvent) => {
        if (event.data && event.data.size > 0) {
          mediaChunksRef.current.push(event.data);
          // Also stream chunks over WebSocket for real-time server transcription
          if (wsRef.current && wsConnectedRef.current) {
            event.data.arrayBuffer().then((buf) => {
              if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
                wsRef.current.send(buf);
              }
            }).catch(() => { /* ignore */ });
          }
        }
      };

      // Smaller timeslice for faster chunk delivery (100ms instead of 250ms)
      recorder.start(100);
      mediaRecorderRef.current = recorder;

      // Open WebSocket connection for streaming transcription
      connectTranscribeWs();

      return true;
    } catch (err) {
      const messageText = err instanceof Error ? err.message : "Microphone permission denied.";
      setError(`Voice error: ${messageText}`);
      return false;
    }
  }

  async function stopAudioCapture(): Promise<Blob | null> {
    if (pendingRecorderStopRef.current) {
      return pendingRecorderStopRef.current;
    }

    const recorder = mediaRecorderRef.current;
    if (!recorder) {
      return null;
    }

    const finalizeBlob = () => {
      const chunkCount = mediaChunksRef.current.length;
      const mimeType = recorder.mimeType || "audio/webm";
      const blob = chunkCount > 0 ? new Blob(mediaChunksRef.current, { type: mimeType }) : null;
      mediaChunksRef.current = [];
      mediaRecorderRef.current = null;
      // Intentionally DO NOT stop the mediaStreamRef tracks!
      // Keeping the tracks active prevents the 1-2 second hardware warmup lag on subsequent recordings.
      return blob;
    };

    if (recorder.state === "inactive") {
      return finalizeBlob();
    }

    pendingRecorderStopRef.current = new Promise<Blob | null>((resolve) => {
      const onStop = () => {
        resolve(finalizeBlob());
      };
      recorder.addEventListener("stop", onStop, { once: true });
      recorder.stop();
    }).finally(() => {
      pendingRecorderStopRef.current = null;
    });

    return pendingRecorderStopRef.current;
  }

  async function finalizeVoiceCapture(recognizedText: string) {
    let finalText = recognizedText.trim();
    const recognizedWordCount = finalText ? finalText.split(/\s+/).filter(Boolean).length : 0;

    const recordedBlob = await stopAudioCapture();
    const shouldUseServerTranscription = Boolean(recordedBlob) && recognizedWordCount === 0;

    if (shouldUseServerTranscription && recordedBlob) {
      // Try WebSocket finalize first for faster result, fall back to HTTP
      let wsTranscriptReceived = false;

      if (wsRef.current && wsConnectedRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        try {
          setStatus("Transcribing audio");
          const wsResult = await new Promise<string>((resolve, reject) => {
            const ws = wsRef.current;
            if (!ws || ws.readyState !== WebSocket.OPEN) {
              reject(new Error("WS not open"));
              return;
            }
            const timeout = setTimeout(() => reject(new Error("WS timeout")), 8000);
            const handler = (ev: MessageEvent) => {
              clearTimeout(timeout);
              ws.removeEventListener("message", handler);
              try {
                const data = JSON.parse(ev.data);
                if (data.error) reject(new Error(data.error));
                else resolve(data.text || "");
              } catch { resolve(""); }
            };
            ws.addEventListener("message", handler);
            ws.send(JSON.stringify({ action: "finalize", language: navigator.language || "en-US" }));
          });
          if (wsResult && (!finalText || wsResult.length > finalText.length + 12)) {
            finalText = wsResult;
          }
          wsTranscriptReceived = true;
        } catch {
          // WS failed, fall through to HTTP
        }
      }

      if (!wsTranscriptReceived) {
        const previousStatus = status;
        try {
          setStatus("Transcribing audio");
          const serverTranscript = await transcribeAudioBlob(recordedBlob);
          if (serverTranscript && (!finalText || serverTranscript.length > finalText.length + 12)) {
            finalText = serverTranscript;
          }
        } catch {
          // Keep browser transcript if server fallback fails.
        } finally {
          setStatus(previousStatus || "Idle");
        }
      }
    }

    finalText = finalText.replace(/\s+/g, " ").trim();
    transcriptBufferRef.current = finalText;
    interimTranscriptRef.current = "";
    setTranscript(finalText);
    setMessage(finalText);

    if (finalText && autoSendRef.current) {
      void handleSend(finalText, "voice");
    } else if (finalText) {
      briefRef.current?.focus();
      setToast("Transcript ready - edit or click Send Brief");
    } else {
      setError("I couldn't catch that clearly. Please try again.");
    }
  }

  function connectTranscribeWs() {
    // Reuse existing open connection
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) return;
    // Close stale connection
    if (wsRef.current) {
      try { wsRef.current.close(); } catch { /* ignore */ }
    }
    try {
      const ws = new WebSocket(wsUrl);
      ws.binaryType = "arraybuffer";
      ws.onopen = () => { wsConnectedRef.current = true; };
      ws.onclose = () => { wsConnectedRef.current = false; };
      ws.onerror = () => { wsConnectedRef.current = false; };
      wsRef.current = ws;
    } catch {
      // WebSocket not available — will fall back to HTTP POST
      wsConnectedRef.current = false;
    }
  }

  function startVoice() {
    voiceModeActiveRef.current = true;
    manualStopRef.current = false;
    if (restartTimerRef.current !== null) {
      window.clearTimeout(restartTimerRef.current);
      restartTimerRef.current = null;
    }
    if (silenceTimerRef.current !== null) {
      window.clearTimeout(silenceTimerRef.current);
      silenceTimerRef.current = null;
    }
    if (window.speechSynthesis) {
      window.speechSynthesis.cancel();
    }
    if (recognitionRef.current) {
      return;
    }
    void startAudioCapture();
    const SpeechRecognitionApi = window.SpeechRecognition ?? window.webkitSpeechRecognition;
    if (!SpeechRecognitionApi) {
      setListening(true);
      setStatus("Listening");
      return;
    }

    setError("");
    // Only clear the transcript buffer on a fresh start (not a mid-hold restart)
    if (!spacebarHeldRef.current) {
      setTranscript("");
      transcriptBufferRef.current = "";
      interimTranscriptRef.current = "";
    }

    const recognition = new SpeechRecognitionApi();
    recognition.lang = navigator.language || "en-US";
    recognition.continuous = true;   // Keep listening through pauses
    recognition.interimResults = true;
    recognition.maxAlternatives = 3;

    recognition.onstart = () => {
      setListening(true);
      setStatus("Listening");
      lastSpeechTimeRef.current = Date.now();
    };

    recognition.onresult = (event: SpeechRecognitionEvent) => {
      let interim = "";
      let gotFinal = false;
      for (let i = event.resultIndex; i < event.results.length; i += 1) {
        const result = event.results[i];
        const chunk = result[0]?.transcript ?? "";
        if (result.isFinal) {
          gotFinal = true;
          if (chunk.trim()) {
            const merged = `${transcriptBufferRef.current} ${chunk}`.replace(/\s+/g, " ").trim();
            transcriptBufferRef.current = merged;
          }
        } else {
          interim += `${chunk} `;
        }
      }
      interimTranscriptRef.current = interim.replace(/\s+/g, " ").trim();
      const liveTranscript = `${transcriptBufferRef.current} ${interimTranscriptRef.current}`.replace(/\s+/g, " ").trim();
      setTranscript(liveTranscript);
      setMessage(liveTranscript);

      // Reset silence timer whenever speech is detected
      if (gotFinal || interim.trim()) {
        lastSpeechTimeRef.current = Date.now();
        if (silenceTimerRef.current !== null) {
          window.clearTimeout(silenceTimerRef.current);
        }
        
        // Final fallback timeout for pure silence
        if (!spacebarHeldRef.current) {
          silenceTimerRef.current = window.setTimeout(() => {
            silenceTimerRef.current = null;
            if (voiceModeActiveRef.current && !spacebarHeldRef.current && transcriptBufferRef.current.trim()) {
              stopVoice();
            }
          }, SILENCE_TIMEOUT_MS);
        }
      }
    };

    recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
      // "no-speech" and "aborted" are expected during hold — don't treat as errors
      if (event.error === "no-speech" || event.error === "aborted") return;
      setListening(false);
      recognitionRef.current = null;
      voiceModeActiveRef.current = false;
      setError(`Voice error: ${event.error}`);
    };

    recognition.onend = () => {
      recognitionRef.current = null;

      // If mic mode is still active (space-hold or mic toggle), the browser may have ended
      // this segment automatically. Restart and keep accumulating transcript.
      if (!manualStopRef.current && (spacebarHeldRef.current || voiceModeActiveRef.current) && !busy && !isSpeaking) {
        recognitionRef.current = null;
        restartTimerRef.current = window.setTimeout(() => {
          restartTimerRef.current = null;
          startVoice();
        }, 80); // Reduced from 120ms for faster restart
        return;
      }

      // Spacebar was released — finalize and submit
      setListening(false);
      setStatus("Idle");
      const finalText = `${transcriptBufferRef.current} ${interimTranscriptRef.current}`.replace(/\s+/g, " ").trim();
      void finalizeVoiceCapture(finalText);
    };

    recognitionRef.current = recognition;
    try {
      recognition.start();
    } catch {
      recognitionRef.current = null;
      setListening(false);
      setError("Could not start voice capture. Please try again.");
    }
  }

  function stopVoice() {
    voiceModeActiveRef.current = false;
    manualStopRef.current = true;
    if (restartTimerRef.current !== null) {
      window.clearTimeout(restartTimerRef.current);
      restartTimerRef.current = null;
    }
    if (silenceTimerRef.current !== null) {
      window.clearTimeout(silenceTimerRef.current);
      silenceTimerRef.current = null;
    }
    if (window.speechSynthesis) {
      window.speechSynthesis.cancel();
    }
    setStatus("Idle");
    const hasRecognitionSession = Boolean(recognitionRef.current);
    recognitionRef.current?.stop();
    if (!hasRecognitionSession) {
      setListening(false);
      void finalizeVoiceCapture("");
    }
  }

  const visualState: VisualState = isSpeaking ? "speaking" : busy ? "thinking" : listening ? "listening" : status === "Transcribing" ? "thinking" : "idle";

  async function openPaperCanvas() {
    // Try direct protocol link first — no backend round-trip needed
    try {
      window.open("paper://", "_self");
      setStatus("Paper Canvas opening...");
      return;
    } catch {
      // Protocol not installed or blocked — fall through to backend launcher
    }

    setStatus("Opening Paper Canvas...");
    setError("");

    const desktopUri = process.env.NEXT_PUBLIC_PAPER_DESKTOP_URI;
    if (desktopUri) {
      window.location.href = desktopUri;
      return;
    }

    try {
      const response = await fetch(`${api}/paper/open`, { method: "POST" });
      const payload = (await response.json()) as PaperOpenResponse;
      setStatus(payload.message || (payload.opened ? "Paper launch command sent." : "Unable to launch Paper."));
      if (!payload.opened) {
        setError(payload.message);
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Failed to call /paper/open";
      setError(msg);
      setStatus("Paper launch unavailable");
    }
  }

  return (
    <main className={styles.page}>
      <header className={styles.topBar}>
        <div className={styles.topLeft}>
          <div className={styles.brand}>Vibeframe</div>
          <div className={styles.connected}>PAPER CONNECTED</div>
        </div>
        <div className={styles.topRight}>
          <span className={styles.signal}>{status || (listening ? "Listening" : "Idle")}</span>
          <button className={styles.navButton} type="button" onClick={openPaperCanvas}>
            Paper Canvas
          </button>
        </div>
      </header>

      <section className={styles.content}>
        <div className={styles.leftPanel}>
          <section className={styles.voiceZone}>
            <div className={styles.voiceZoneInner} data-state={visualState}>
              <WaveformVisualizer state={visualState} />
              <p className={styles.voiceHint}>
                {visualState === "listening"
                  ? "Listening..."
                  : visualState === "thinking"
                    ? status === "Transcribing"
                      ? "Transcribing your voice..."
                      : "Agents thinking..."
                    : visualState === "speaking"
                      ? "Vibeframe speaking"
                      : "Hold Space · Click Mic · or Type below"}
              </p>
            </div>
          </section>

          <section className={styles.inputZone}>
            <div className={styles.inputHeader}>
              <span className={styles.sectionLabel}>Brief</span>
              <label className={styles.toggleRow}>
                <span className={styles.toggleLabel}>Auto-send voice</span>
                <span className={styles.toggleSwitch} data-active={autoSend ? "true" : "false"}>
                  <input
                    aria-label="Auto-send voice"
                    checked={autoSend}
                    type="checkbox"
                    onChange={(event) => setAutoSend(event.target.checked)}
                  />
                  <span className={styles.toggleTrack}>
                    <span className={styles.toggleThumb} />
                  </span>
                </span>
              </label>
            </div>

            <textarea
              ref={briefRef}
              className={styles.textarea}
              value={message}
              onChange={(event) => {
                setMessage(event.target.value);
                transcriptBufferRef.current = event.target.value;
                setTranscript(event.target.value);
              }}
              placeholder="Describe your website or hold Space to speak..."
              rows={4}
            />

            <div className={styles.inputActions}>
              <button
                className={`${styles.micButton} ${listening ? styles.micButtonActive : ""}`}
                type="button"
                onClick={() => {
                  if (listening) {
                    stopVoice();
                  } else {
                    startVoice();
                  }
                }}
                disabled={!voiceSupported && !listening}
                aria-label={listening ? "Stop recording" : "Start recording"}
              >
                {listening ? (
                  <svg viewBox="0 0 24 24" aria-hidden="true">
                    <rect x="7" y="7" width="10" height="10" rx="3" />
                  </svg>
                ) : (
                  <svg viewBox="0 0 24 24" aria-hidden="true">
                    <path d="M12 14a3 3 0 0 0 3-3V7a3 3 0 0 0-6 0v4a3 3 0 0 0 3 3Z" />
                    <path d="M5 11a7 7 0 0 0 14 0" />
                    <path d="M12 18v3" />
                  </svg>
                )}
              </button>

              <div className={styles.actionSpacer} />

              <button className={styles.sendButton} disabled={busy} onClick={() => void handleSend(message, "text")} type="button">
                Send Brief
              </button>
            </div>
          </section>

        </div>

        <section
          className={styles.rightPanel}
          id="right-panel"
          style={{ "--chat-pane-size": `${chatPaneSize}%` } as CSSProperties}
        >
          <div className={styles.historyCard}>
            <div className={styles.historyHeader}>
              <span className={styles.logsTitle}>CONVERSATION</span>
            </div>
            <section className={styles.historyZoneRight}>
              {chat.length ? (
                chat.map((item) => {
                  const isUser = item.role === "user";
                  return (
                    <div className={`${styles.historyLine} ${isUser ? styles.historyLineUser : styles.historyLineAi}`} key={item.id}>
                      <span className={isUser ? styles.historyPrefixUser : styles.historyPrefixAi}>{isUser ? "You" : "AI"}</span>
                      <span className={`${styles.historyText} ${isUser ? styles.historyTextUser : styles.historyTextAi}`}>{item.text}</span>
                    </div>
                  );
                })
              ) : (
                <div className={styles.historyEmpty}>Conversation history will appear here.</div>
              )}
              <div ref={chatEndRef} />
            </section>
          </div>

          <div
            className={styles.logsDivider}
            role="separator"
            aria-label="Resize conversation and logs"
            aria-orientation="horizontal"
            onMouseDown={() => {
              isResizingRef.current = true;
              document.body.style.cursor = "row-resize";
              document.body.style.userSelect = "none";
            }}
          />

          <div className={`${styles.logsCard} ${logsMinimized ? styles.logsCardMinimized : ""}`}>
            <div className={styles.logsHeader}>
              <span className={styles.logsTitle}>AGENT LOGS</span>
              <button
                type="button"
                onClick={() => setLogsMinimized((value) => !value)}
                className={styles.minimizeButton}
                aria-label={logsMinimized ? "Expand logs" : "Minimize logs"}
              >
                {logsMinimized ? "+" : "−"}
              </button>
            </div>

            {!logsMinimized ? (
              <div className={styles.logsList}>
                {events.length ? (
                  events.map((event) => {
                    const color = getAgentColor(event.agent);
                    const label = getAgentLabel(event.agent);
                    return (
                      <div className={styles.logRow} key={event.id}>
                        <span className={styles.logDot} style={{ backgroundColor: color }} />
                        <span className={styles.logAgent} style={{ color }}>
                          {label}
                        </span>
                        <span className={styles.logMessage}>{event.message}</span>
                        <span className={styles.logTime}>{formatTime(event.timestamp)}</span>
                      </div>
                    );
                  })
                ) : (
                  <div className={styles.logsEmpty}>Waiting for agent activity...</div>
                )}
              </div>
            ) : null}
          </div>

          <div className={styles.roundCard}>
            <span className={styles.roundLabel}>Round</span>
            <div className={styles.roundDots} aria-hidden="true">
              <span className={round >= 1 ? styles.roundDotActive : styles.roundDot} />
              <span className={round >= 2 ? styles.roundDotActive : styles.roundDot} />
              <span className={round >= 3 ? styles.roundDotActive : styles.roundDot} />
            </div>
            <span className={styles.roundAction}>{busy ? "Designer working..." : status}</span>
            <button className={styles.stopButton} onClick={resetSession} type="button">
              Stop
            </button>
          </div>
        </section>
      </section>

      {error ? <div className={styles.errorBanner}>{error}</div> : null}

      {toastVisible ? <div className={styles.toast}>{toast}</div> : null}
    </main>
  );
}

"use client";

import { useEffect, useMemo, useRef, useState } from "react";
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

export default function HomePage() {
  const api = useMemo(() => backendUrl(), []);
  const [conversationId, setConversationId] = useState(() => newConversationId());

  const [message, setMessage] = useState("");
  const [transcript, setTranscript] = useState("");
  const [assistantText, setAssistantText] = useState("Awaiting your brief...");
  const [stage, setStage] = useState("intake");
  const [questions, setQuestions] = useState<string[]>([]);
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const logsEndRef = useRef<HTMLDivElement>(null);
  const [chat, setChat] = useState<ChatTurn[]>([]);
  const chatEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [chat]);
  const [busy, setBusy] = useState(false);
  const [listening, setListening] = useState(false);
  const [voiceSupported, setVoiceSupported] = useState(false);
  const [status, setStatus] = useState("Ready");
  const [error, setError] = useState("");
  const [round, setRound] = useState(0);

  const [logsMinimized, setLogsMinimized] = useState(false);
  const [logsHeight, setLogsHeight] = useState(240);
  const dragRef = useRef(false);

  useEffect(() => {
    const handleMouseUp = () => {
      dragRef.current = false;
      document.body.style.cursor = "default";
    };
    const handleMouseMove = (e: MouseEvent) => {
      if (!dragRef.current) return;
      // dragging up (negative movementY) increases height 
      setLogsHeight((h) => Math.max(80, Math.min(800, h - e.movementY)));
    };
    document.addEventListener("mouseup", handleMouseUp);
    document.addEventListener("mousemove", handleMouseMove);
    return () => {
      document.removeEventListener("mouseup", handleMouseUp);
      document.removeEventListener("mousemove", handleMouseMove);
    };
  }, []);

  const recognitionRef = useRef<SpeechRecognition | null>(null);
  const transcriptBufferRef = useRef("");
  const voiceModeActiveRef = useRef(false);
  const spacebarHeldRef = useRef(false);

  useEffect(() => {
    if (events.length > 0) {
      logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [events]);

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
    const SpeechRecognitionApi = window.SpeechRecognition ?? window.webkitSpeechRecognition;
    setVoiceSupported(Boolean(SpeechRecognitionApi));
  }, []);

  useEffect(() => {
    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
    };
  }, []);

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
    if (window.speechSynthesis) {
      window.speechSynthesis.cancel();
    }
    if (recognitionRef.current) {
      recognitionRef.current.stop();
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
    setAssistantText("Awaiting your brief...");
    setChat([]);
  }

  function speak(text: string) {
    if (!text || !window.speechSynthesis) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "en-US";
    utterance.onend = () => {
      if (voiceModeActiveRef.current && !busy) {
        startVoice();
      }
    };
    window.speechSynthesis.speak(utterance);
  }

  async function submitGenerate(text: string, source: "text" | "voice") {
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
    // Clear the voice transcript display immediately after sending
    if (source === "voice") {
      setTranscript("");
      transcriptBufferRef.current = "";
    } else {
      setMessage("");
    }

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
      setAssistantText(msg);
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

  function startVoice() {
    voiceModeActiveRef.current = true;
    if (window.speechSynthesis) {
      window.speechSynthesis.cancel();
    }
    const SpeechRecognitionApi = window.SpeechRecognition ?? window.webkitSpeechRecognition;
    if (!SpeechRecognitionApi) {
      setError("Voice capture requires Chrome Web Speech API.");
      return;
    }

    setError("");
    // Only clear the transcript buffer on a fresh start (not a mid-hold restart)
    if (!spacebarHeldRef.current) {
      setTranscript("");
      transcriptBufferRef.current = "";
    }

    const recognition = new SpeechRecognitionApi();
    recognition.lang = "en-US";
    recognition.continuous = true;   // Keep listening through pauses
    recognition.interimResults = true;
    recognition.maxAlternatives = 1;

    recognition.onstart = () => setListening(true);

    recognition.onresult = (event: SpeechRecognitionEvent) => {
      let interim = "";
      for (let i = event.resultIndex; i < event.results.length; i += 1) {
        const result = event.results[i];
        const chunk = result[0]?.transcript ?? "";
        if (result.isFinal) {
          transcriptBufferRef.current += `${chunk} `;
        } else {
          interim += chunk;
        }
      }
      setTranscript(`${transcriptBufferRef.current}${interim}`.trim());
    };

    recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
      // "no-speech" and "aborted" are expected during hold — don't treat as errors
      if (event.error === "no-speech" || event.error === "aborted") return;
      setListening(false);
      setError(`Voice error: ${event.error}`);
    };

    recognition.onend = () => {
      // If spacebar is still held, the browser cut us off — restart immediately
      if (spacebarHeldRef.current) {
        recognitionRef.current = null;
        startVoice();
        return;
      }
      // Spacebar was released — finalize and submit
      setListening(false);
      const finalText = transcriptBufferRef.current.trim();
      setTranscript(finalText);
      if (finalText) {
        void submitGenerate(finalText, "voice");
      }
    };

    recognitionRef.current = recognition;
    recognition.start();
  }

  function stopVoice() {
    voiceModeActiveRef.current = false;
    if (window.speechSynthesis) {
      window.speechSynthesis.cancel();
    }
    recognitionRef.current?.stop();
  }

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
        <div className={styles.brandBlock}>
          <div className={styles.brand}>Vibeframe</div>
          <div className={styles.connected}>PAPER CONNECTED</div>
        </div>
        <div className={styles.topActions}>
          <span className={styles.signal}>{listening ? "Listening" : "Idle"}</span>
          <button className={styles.primaryButton} type="button" onClick={openPaperCanvas}>
            Paper Canvas
          </button>
        </div>
      </header>

      <section className={styles.content}>
        <aside className={styles.sideRail}>
          <button className={styles.railItem} onClick={resetSession} type="button">
            NEW
          </button>
        </aside>

        <div className={styles.leftPanel}>
          <div className={styles.orbWrap}>
            <div className={styles.orbGlow} />
            <div className={styles.orb}>{listening ? "◉" : "◍"}</div>
          </div>
          <p className={styles.kicker}>{listening ? "LISTENING..." : transcript ? "CURRENT TRANSCRIPTION" : "HOLD SPACE TO TALK"}</p>
          <p className={styles.transcript}>
            {transcript ? (transcript.length > 120 ? `${transcript.slice(0, 120)}...` : transcript) : (listening ? "Listening..." : "Vibeframe Ready")}
          </p>

          <div className={styles.controls}>
            <textarea
              className={styles.input}
              value={message}
              onChange={(event) => setMessage(event.target.value)}
              rows={3}
              placeholder="Describe your website (e.g., 'Modern SaaS landing page for a crypto wallet')..."
            />
            <div className={styles.controlRow}>
              <button className={styles.primaryButton} disabled={busy} onClick={() => void submitGenerate(message, "text")} type="button">
                {busy ? "Sending..." : "Send"}
              </button>
              <button className={styles.secondaryButton} disabled={!voiceSupported || listening} onClick={startVoice} type="button">
                Start Voice
              </button>
              <button className={styles.secondaryButton} disabled={!listening} onClick={stopVoice} type="button">
                Stop
              </button>
            </div>

          </div>
        </div>

        <section className={styles.rightPanel}>
          <div className={styles.streamHeader}>
            <h2>LIVE COORDINATION STREAM</h2>
            <span>ACTIVE SYNC</span>
          </div>

          <div className={styles.feed}>
            {chat.length ? (
              chat.map((item) => (
                <article
                  key={item.id}
                  className={`${styles.bubble} ${
                    item.role === "user" ? styles.bubbleUser : item.role === "assistant" ? styles.bubbleAssistant : styles.bubbleSystem
                  }`}
                >
                  <p>{item.text}</p>
                  <small>{item.meta}</small>
                </article>
              ))
            ) : (
              <div className={styles.feedEmpty}>
                <div className={styles.emptyIcon}>✦</div>
                <div className={styles.emptyText}>
                  Your creation journey starts here.<br />
                  Voice your ideas or type a brief to begin.
                </div>
              </div>
            )}
            <div ref={chatEndRef} />
          </div>

          <div className={styles.logsWrap}>
            <div
              className={styles.logsResizer}
              onMouseDown={() => {
                if (logsMinimized) return;
                dragRef.current = true;
                document.body.style.cursor = "row-resize";
              }}
            >
              <div className={styles.resizerHandle} />
              <div className={styles.logsActions}>
                <span className={styles.logsTitle}>AGENT LOGS</span>
                <button
                  type="button"
                  onClick={() => setLogsMinimized((m) => !m)}
                  className={styles.minimizeBtn}
                  onMouseDown={(e) => e.stopPropagation()}
                >
                  {logsMinimized ? (
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                      <polyline points="18 15 12 9 6 15" />
                    </svg>
                  ) : (
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                      <line x1="5" y1="12" x2="19" y2="12" />
                    </svg>
                  )}
                </button>
              </div>
            </div>

            {!logsMinimized && (
              <div className={styles.logs} style={{ height: logsHeight }}>
                <div className={styles.logsInner}>
                  {events.length ? (
                    events.map((ev) => {
                      const color = AGENT_COLORS[ev.agent] ?? "#6b7280";
                      const label = AGENT_LABELS[ev.agent] ?? ev.agent.replaceAll("_", " ");
                      return (
                        <div 
                          key={ev.id} 
                          className={styles.agentBubble}
                          style={{
                            "--agent-color": color,
                            "--agent-bg-color": hexToRgba(color, 0.08),
                            "--agent-border-color": hexToRgba(color, 0.18),
                          } as React.CSSProperties}
                        >
                          <div className={styles.agentAvatarWrap}>
                            <span className={styles.agentPulse} />
                            <span className={styles.agentDot} />
                          </div>
                          <div className={styles.agentContent}>
                            <span className={styles.agentName} style={{ color }}>
                              {label}
                            </span>
                            <span className={styles.agentMessage}>{ev.message}</span>
                          </div>
                        </div>
                      );
                    })
                  ) : (
                    <div className={styles.logsEmpty}>Waiting for agent activity...</div>
                  )}
                  <div ref={logsEndRef} />
                </div>
              </div>
            )}
          </div>
        </section>
      </section>

      {error ? <div className={styles.errorBanner}>{error}</div> : null}
    </main>
  );
}

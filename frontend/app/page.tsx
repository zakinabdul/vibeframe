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
  message?: string;
  questions?: string[];
  round?: number;
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

function formatEventLine(raw: string) {
  try {
    const parsed = JSON.parse(raw) as StreamEvent;
    const parts = [parsed.type, parsed.provider, parsed.message].filter(Boolean);
    if (Array.isArray(parsed.questions) && parsed.questions.length) {
      parts.push(`${parsed.questions.length} questions`);
    }
    return parts.join(" | ") || raw;
  } catch {
    return raw;
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

  const [message, setMessage] = useState(
    "I want to build a landing page for a boutique hotel in the Swiss Alps with a focus on luxury and minimalism."
  );
  const [transcript, setTranscript] = useState("");
  const [assistantText, setAssistantText] = useState("Share your brief and I will create a palette first, then build after your approval.");
  const [stage, setStage] = useState("intake");
  const [questions, setQuestions] = useState<string[]>([]);
  const [events, setEvents] = useState<string[]>([]);
  const [chat, setChat] = useState<ChatTurn[]>([
    {
      id: "seed",
      role: "assistant",
      text: "How many pages do you need? I will use that to scope navigation and component system.",
      meta: "AI Agent",
    },
  ]);
  const [busy, setBusy] = useState(false);
  const [listening, setListening] = useState(false);
  const [voiceSupported, setVoiceSupported] = useState(false);
  const [status, setStatus] = useState("Ready");
  const [error, setError] = useState("");
  const [round, setRound] = useState(0);

  const recognitionRef = useRef<SpeechRecognition | null>(null);
  const transcriptBufferRef = useRef("");
  const voiceModeActiveRef = useRef(false);

  useEffect(() => {
    const EventSourceImpl = window.EventSource;
    const source = new EventSourceImpl(`${api}/stream`);

    source.onmessage = (event) => {
      if (!event.data) {
        return;
      }
      const line = formatEventLine(event.data);
      setEvents((current) => [line, ...current].slice(0, 30));

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
      setEvents((current) => ["stream_disconnected", ...current].slice(0, 30));
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

  function pushChat(role: ChatTurn["role"], text: string, meta: string) {
    setChat((current) => [
      {
        id: `${role}-${Date.now()}-${Math.random().toString(16).slice(2)}`,
        role,
        text,
        meta,
      },
      ...current,
    ]);
  }

  function resetSession() {
    voiceModeActiveRef.current = false;
    if (window.speechSynthesis) {
      window.speechSynthesis.cancel();
    }
    if (recognitionRef.current) {
      recognitionRef.current.stop();
    }
    setConversationId(newConversationId());
    setStage("intake");
    setQuestions([]);
    setRound(0);
    setError("");
    setTranscript("");
    setStatus("New session");
    setAssistantText("New session ready. I will ask for palette approval before build.");
    setChat([
      {
        id: `reset-${Date.now()}`,
        role: "assistant",
        text: "New session created. Send your new design brief.",
        meta: "AI Agent",
      },
    ]);
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

      if (data.questions?.length) {
        pushChat("system", data.questions.map((item, index) => `${index + 1}. ${item}`).join("\n"), "Clarifying Questions");
      }
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
    setTranscript("");
    transcriptBufferRef.current = "";

    const recognition = new SpeechRecognitionApi();
    recognition.lang = "en-US";
    recognition.continuous = false;
    recognition.interimResults = true;

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
      setListening(false);
      setError(`Voice error: ${event.error}`);
    };

    recognition.onend = () => {
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
    setStatus("Opening Paper Canvas...");
    setError("");

    const desktopUri = process.env.NEXT_PUBLIC_PAPER_DESKTOP_URI;
    if (desktopUri) {
      try {
        window.location.href = desktopUri;
      } catch {
        // Keep trying backend launcher below.
      }
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
          <button className={styles.railItem}>VOICE</button>
          <button className={styles.railItem} onClick={resetSession} type="button">
            NEW
          </button>
        </aside>

        <div className={styles.leftPanel}>
          <div className={styles.orbWrap}>
            <div className={styles.orbGlow} />
            <div className={styles.orb}>{listening ? "◉" : "◍"}</div>
          </div>
          <p className={styles.kicker}>CURRENT TRANSCRIPTION</p>
          <p className={styles.transcript}>{transcript || message}</p>
          <p className={styles.assistant}>{assistantText}</p>

          {questions.length ? (
            <div className={styles.questionTray}>
              {questions.map((question) => (
                <button key={question} className={styles.questionChip} onClick={() => setMessage(question)} type="button">
                  {question}
                </button>
              ))}
            </div>
          ) : null}

          <div className={styles.controls}>
            <textarea
              className={styles.input}
              value={message}
              onChange={(event) => setMessage(event.target.value)}
              rows={3}
              placeholder="Describe the design you want..."
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
            {stage === "palette_requested" ? (
              <button
                className={styles.primaryButton}
                disabled={busy}
                onClick={() => void submitGenerate("Approve Indigo Night and proceed with build.", "text")}
                type="button"
              >
                Approve Palette + Build
              </button>
            ) : null}
          </div>
        </div>

        <section className={styles.rightPanel}>
          <div className={styles.streamHeader}>
            <h2>LIVE COORDINATION STREAM</h2>
            <span>ACTIVE SYNC</span>
          </div>

          <div className={styles.feed}>
            {chat.map((item) => (
              <article
                key={item.id}
                className={`${styles.bubble} ${
                  item.role === "user" ? styles.bubbleUser : item.role === "assistant" ? styles.bubbleAssistant : styles.bubbleSystem
                }`}
              >
                <p>{item.text}</p>
                <small>{item.meta}</small>
              </article>
            ))}
          </div>

          <div className={styles.logs}>
            {events.length ? (
              events.map((line, idx) => (
                <div key={`${idx}-${line}`} className={styles.logLine}>
                  {line}
                </div>
              ))
            ) : (
              <div className={styles.logLine}>No stream events yet.</div>
            )}
          </div>
        </section>
      </section>

      <footer className={styles.footer}>
        <div>PROGRESS: ROUND {round || 1} OF 3</div>
        <div>{status}</div>
      </footer>

      {error ? <div className={styles.errorBanner}>{error}</div> : null}
    </main>
  );
}

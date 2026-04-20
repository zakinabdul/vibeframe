"use client";

import { useEffect, useState } from "react";

export type WaveformState = "idle" | "listening" | "thinking" | "speaking";

type WaveformVisualizerProps = {
  state: WaveformState;
};

const BAR_COUNT = 28;
const IDLE_HEIGHT = 4;

function randomHeight(min: number, max: number) {
  return Math.round(min + Math.random() * (max - min));
}

export default function WaveformVisualizer({ state }: WaveformVisualizerProps) {
  const [barHeights, setBarHeights] = useState<number[]>(() => Array.from({ length: BAR_COUNT }, () => IDLE_HEIGHT));

  useEffect(() => {
    if (state === "idle") {
      setBarHeights(Array.from({ length: BAR_COUNT }, () => IDLE_HEIGHT));
      return;
    }

    if (state === "listening") {
      setBarHeights(Array.from({ length: BAR_COUNT }, () => randomHeight(8, 64)));
      const timer = window.setInterval(() => {
        setBarHeights(Array.from({ length: BAR_COUNT }, () => randomHeight(8, 64)));
      }, 120);
      return () => window.clearInterval(timer);
    }

    if (state === "speaking") {
      setBarHeights(Array.from({ length: BAR_COUNT }, () => randomHeight(16, 56)));
      const timer = window.setInterval(() => {
        setBarHeights(Array.from({ length: BAR_COUNT }, () => randomHeight(16, 56)));
      }, 80);
      return () => window.clearInterval(timer);
    }

    let tick = 0;
    const timer = window.setInterval(() => {
      tick += 1;
      setBarHeights(
        Array.from({ length: BAR_COUNT }, (_, index) => {
          const phase = tick / 6 + index * 0.42;
          const wave = Math.sin(phase) * 10 + Math.sin(phase * 0.55 + index * 0.18) * 5;
          return Math.round(24 + wave);
        }),
      );
    }, 80);

    return () => window.clearInterval(timer);
  }, [state]);

  const barColor = state === "thinking" ? "#a78bfa" : state === "speaking" ? "#22c55e" : "#6366f1";
  const barOpacity = state === "idle" ? 0.25 : 1;
  const barShadow =
    state === "listening"
      ? "0 0 8px rgba(99, 102, 241, 0.38)"
      : state === "thinking"
        ? "0 0 10px rgba(167, 139, 250, 0.24)"
        : state === "speaking"
          ? "0 0 8px rgba(34, 197, 94, 0.28)"
          : "none";

  return (
    <div
      aria-hidden="true"
      style={{
        display: "flex",
        alignItems: "flex-end",
        gap: 3,
        height: 72,
      }}
    >
      {barHeights.map((height, index) => (
        <div
          key={index}
          style={{
            width: 3,
            height,
            minHeight: IDLE_HEIGHT,
            maxHeight: 64,
            borderRadius: 999,
            backgroundColor: barColor,
            opacity: barOpacity,
            boxShadow: barShadow,
            transition: "height 80ms ease, background-color 120ms ease, box-shadow 120ms ease, opacity 120ms ease",
          }}
        />
      ))}
    </div>
  );
}

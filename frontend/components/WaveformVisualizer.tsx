"use client";

import React, { useEffect, useRef, useState } from "react";
import { Activity, Volume2 } from "lucide-react";

interface WaveformVisualizerProps {
  file: File | null;
  audioUrl?: string | null;
  label?: string;
  variant?: "emerald" | "cyan";
}

export function WaveformVisualizer({
  file,
  audioUrl,
  label = "Audio Signal Waveform",
  variant = "cyan",
}: WaveformVisualizerProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [duration, setDuration] = useState<number | null>(null);
  const [sampleRate, setSampleRate] = useState<number | null>(null);
  const [channels, setChannels] = useState<number | null>(null);
  const [isDecoding, setIsDecoding] = useState<boolean>(false);
  const [decodeError, setDecodeError] = useState<string | null>(null);

  useEffect(() => {
    if (!file && !audioUrl) {
      setDuration(null);
      setSampleRate(null);
      setChannels(null);
      const canvas = canvasRef.current;
      if (canvas) {
        const ctx = canvas.getContext("2d");
        if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
      }
      return;
    }

    let isCancelled = false;
    setIsDecoding(true);
    setDecodeError(null);

    async function decodeAndRender() {
      try {
        let arrayBuffer: ArrayBuffer;
        if (file) {
          arrayBuffer = await file.arrayBuffer();
        } else if (audioUrl) {
          const res = await fetch(audioUrl);
          arrayBuffer = await res.arrayBuffer();
        } else {
          return;
        }

        const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
        const audioCtx = new AudioCtx();
        const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);

        if (isCancelled) {
          await audioCtx.close();
          return;
        }

        setDuration(audioBuffer.duration);
        setSampleRate(audioBuffer.sampleRate);
        setChannels(audioBuffer.numberOfChannels);

        // Extract peaks from channel 0
        const rawData = audioBuffer.getChannelData(0);
        const canvas = canvasRef.current;
        if (!canvas) {
          await audioCtx.close();
          return;
        }

        const ctx = canvas.getContext("2d");
        if (!ctx) {
          await audioCtx.close();
          return;
        }

        // Adjust for high-DPI displays
        const width = canvas.clientWidth * window.devicePixelRatio;
        const height = canvas.clientHeight * window.devicePixelRatio;
        canvas.width = width;
        canvas.height = height;

        ctx.clearRect(0, 0, width, height);

        // Draw graticule grid
        ctx.strokeStyle = "rgba(30, 41, 59, 0.4)";
        ctx.lineWidth = 1;
        const gridStep = width / 12;
        for (let x = 0; x <= width; x += gridStep) {
          ctx.beginPath();
          ctx.moveTo(x, 0);
          ctx.lineTo(x, height);
          ctx.stroke();
        }
        ctx.beginPath();
        ctx.moveTo(0, height / 2);
        ctx.lineTo(width, height / 2);
        ctx.stroke();

        // Calculate peak bars
        const numBars = Math.floor(canvas.clientWidth / 3.5);
        const blockSize = Math.floor(rawData.length / numBars);
        const peaks: number[] = [];

        for (let i = 0; i < numBars; i++) {
          const start = i * blockSize;
          let max = 0;
          for (let j = 0; j < blockSize; j++) {
            const val = Math.abs(rawData[start + j] || 0);
            if (val > max) max = val;
          }
          peaks.push(max);
        }

        // Waveform styling gradient
        const barWidth = (width / numBars) * 0.75;
        const barGap = (width / numBars) * 0.25;
        const halfHeight = height / 2;

        const grad = ctx.createLinearGradient(0, 0, 0, height);
        if (variant === "emerald") {
          grad.addColorStop(0, "#34d399");
          grad.addColorStop(0.5, "#10b981");
          grad.addColorStop(1, "#059669");
          ctx.shadowColor = "rgba(16, 185, 129, 0.4)";
        } else {
          grad.addColorStop(0, "#38bdf8");
          grad.addColorStop(0.5, "#06b6d4");
          grad.addColorStop(1, "#0ea5e9");
          ctx.shadowColor = "rgba(6, 182, 212, 0.4)";
        }
        ctx.shadowBlur = 4;
        ctx.fillStyle = grad;

        for (let i = 0; i < numBars; i++) {
          const peak = Math.min(1.0, peaks[i] || 0);
          const barH = Math.max(3, peak * (halfHeight - 6));
          const x = i * (barWidth + barGap);

          // Symmetrical dual bar from center
          ctx.beginPath();
          ctx.roundRect(x, halfHeight - barH, barWidth, barH * 2, 2);
          ctx.fill();
        }

        await audioCtx.close();
      } catch {
        if (!isCancelled) {
          setDecodeError("Audio waveform preview unavailable (unsupported format or browser codec).");
        }
      } finally {
        if (!isCancelled) setIsDecoding(false);
      }
    }

    decodeAndRender();

    return () => {
      isCancelled = true;
    };
  }, [file, audioUrl, variant]);

  const formatSeconds = (sec: number) => {
    const mins = Math.floor(sec / 60);
    const secs = (sec % 60).toFixed(2);
    return `${mins}:${Number(secs) < 10 ? "0" : ""}${secs}`;
  };

  return (
    <div className="w-full bg-[#0a0f1a] rounded-lg border border-slate-800 p-3 shadow-inner">
      <div className="flex items-center justify-between text-xs text-slate-400 mb-2 font-mono">
        <div className="flex items-center gap-1.5">
          <Activity className={`w-3.5 h-3.5 ${variant === "emerald" ? "text-emerald-400" : "text-cyan-400"}`} />
          <span>{label}</span>
        </div>
        {duration !== null && (
          <div className="flex items-center gap-3 text-slate-300">
            <span>{channels === 1 ? "Mono" : channels === 2 ? "Stereo" : `${channels}ch`}</span>
            <span>{sampleRate ? `${Math.round(sampleRate / 1000)} kHz` : ""}</span>
            <span className="text-emerald-400 font-semibold">{formatSeconds(duration)}</span>
          </div>
        )}
      </div>

      <div className="relative w-full h-24 rounded bg-[#070b12] border border-slate-900 overflow-hidden flex items-center justify-center">
        <canvas
          ref={canvasRef}
          className="w-full h-full block"
          aria-label="Audio waveform canvas preview"
        />

        {isDecoding && (
          <div className="absolute inset-0 bg-[#070b12]/80 flex items-center justify-center text-xs text-slate-400 font-mono gap-2">
            <div className="w-3 h-3 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin" />
            Analyzing waveform...
          </div>
        )}

        {decodeError && (
          <div className="absolute inset-0 bg-[#070b12]/90 flex items-center justify-center p-2 text-xs text-slate-500 font-mono text-center">
            {decodeError}
          </div>
        )}

        {!file && !audioUrl && !isDecoding && (
          <div className="absolute inset-0 flex items-center justify-center text-xs text-slate-600 font-mono gap-2">
            <Volume2 className="w-4 h-4 opacity-50" />
            <span>Upload audio to view frequency waveform</span>
          </div>
        )}
      </div>
    </div>
  );
}

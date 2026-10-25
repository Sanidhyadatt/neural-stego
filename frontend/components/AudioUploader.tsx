"use client";

import React, { useRef, useState } from "react";
import { FileAudio, UploadCloud, X } from "lucide-react";
import { WaveformVisualizer } from "./WaveformVisualizer";

const MAX_BYTES = 25 * 1024 * 1024;
const ALLOWED_EXTS = [".wav", ".mp3", ".flac", ".ogg"];

interface AudioUploaderProps {
  label: string;
  file: File | null;
  onFileSelect: (file: File | null) => void;
  onError: (errorMsg: string) => void;
  accent?: "cyan" | "emerald";
  id?: string;
}

export function AudioUploader({
  label,
  file,
  onFileSelect,
  onError,
  accent = "cyan",
  id,
}: AudioUploaderProps) {
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const validateAndSetFile = (candidate: File) => {
    if (candidate.size > MAX_BYTES) {
      onError(`File exceeds 25 MB limit (${(candidate.size / (1024 * 1024)).toFixed(1)} MB). Please select a smaller audio file.`);
      return;
    }

    const name = candidate.name.toLowerCase();
    const hasValidExt = ALLOWED_EXTS.some((ext) => name.endsWith(ext));
    if (!hasValidExt) {
      onError(`Unsupported format. Please upload WAV, MP3, FLAC, or OGG.`);
      return;
    }

    onFileSelect(candidate);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      validateAndSetFile(e.dataTransfer.files[0]);
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      validateAndSetFile(e.target.files[0]);
    }
    e.target.value = "";
  };

  const handleQuickLoad = async (url: string, name: string, type: string) => {
    try {
      const res = await fetch(url);
      const blob = await res.blob();
      const loadedFile = new File([blob], name, { type });
      validateAndSetFile(loadedFile);
    } catch {
      onError(`Failed to load ${name}.`);
    }
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <label className="block text-xs font-mono font-medium text-slate-300 uppercase tracking-wider">
          {label}
        </label>

        {/* Quick test files loader */}
        <div className="flex flex-wrap items-center gap-1.5 text-[11px] font-mono text-slate-400">
          <span className="text-slate-500 text-[10px]">Test files:</span>
          <button
            type="button"
            id={id ? `${id}-sample-wav-btn` : "sample-wav-btn"}
            onClick={() => handleQuickLoad("/test_files/sample.wav", "sample.wav", "audio/wav")}
            className="px-2 py-0.5 rounded bg-slate-800/80 hover:bg-cyan-950/80 hover:text-cyan-300 border border-slate-700/60 hover:border-cyan-500/40 text-slate-300 transition-colors"
          >
            sample.wav
          </button>
          {id === "reveal-file-input" && (
            <button
              type="button"
              id="reveal-stego-wav-btn"
              onClick={() => handleQuickLoad("/test_files/stego.wav", "stego.wav", "audio/wav")}
              className="px-2 py-0.5 rounded bg-emerald-950/80 hover:bg-emerald-900 border border-emerald-500/40 text-emerald-300 font-semibold transition-colors"
            >
              stego.wav (Encoded)
            </button>
          )}
          <button
            type="button"
            id={id ? `${id}-large-wav-btn` : "large-wav-btn"}
            onClick={() => handleQuickLoad("/test_files/large_30mb.wav", "large_30mb.wav", "audio/wav")}
            className="px-2 py-0.5 rounded bg-slate-800/80 hover:bg-rose-950/80 hover:text-rose-300 border border-slate-700/60 hover:border-rose-500/40 text-slate-300 transition-colors"
          >
            large_30mb.wav (30MB)
          </button>
          <button
            type="button"
            id={id ? `${id}-sample-txt-btn` : "sample-txt-btn"}
            onClick={() => handleQuickLoad("/test_files/sample.txt", "sample.txt", "text/plain")}
            className="px-2 py-0.5 rounded bg-slate-800/80 hover:bg-amber-950/80 hover:text-amber-300 border border-slate-700/60 hover:border-amber-500/40 text-slate-300 transition-colors"
          >
            sample.txt
          </button>
        </div>
      </div>

      {!file ? (
        <div
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onClick={() => inputRef.current?.click()}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") {
              e.preventDefault();
              inputRef.current?.click();
            }
          }}
          tabIndex={0}
          role="button"
          aria-label={`${label}: click or drag and drop audio file`}
          className={`relative border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all ${
            isDragging
              ? "border-cyan-400 bg-cyan-950/20 shadow-lg shadow-cyan-950/50"
              : "border-slate-800 bg-[#090e18]/80 hover:border-slate-700 hover:bg-[#0c1322]"
          } focus-visible:ring-2 focus-visible:ring-cyan-400`}
        >
          <input
            ref={inputRef}
            id={id || "audio-file-input"}
            aria-label="Upload audio file"
            type="file"
            accept=".wav,.mp3,.flac,.ogg,audio/*"
            onChange={handleInputChange}
            className="sr-only"
            tabIndex={-1}
          />
          <div className="flex flex-col items-center justify-center space-y-3">
            <div className="w-12 h-12 rounded-full bg-slate-900/80 border border-slate-800 flex items-center justify-center text-cyan-400 shadow-inner">
              <UploadCloud className="w-6 h-6 animate-pulse" />
            </div>
            <div>
              <p className="text-sm font-medium text-slate-200">
                <span className="text-cyan-400 font-semibold underline underline-offset-2">Click to browse</span> or drag audio file here
              </p>
              <p className="text-xs text-slate-400 mt-1 font-mono">
                WAV, MP3, FLAC, OGG • Max 25 MB (up to 5 min)
              </p>
            </div>
          </div>
        </div>
      ) : (
        <div className="space-y-3">
          <div className="flex items-center justify-between p-3.5 bg-[#0d1424] border border-slate-800 rounded-xl shadow-md">
            <div className="flex items-center space-x-3 overflow-hidden">
              <div className="w-10 h-10 rounded-lg bg-slate-900 border border-slate-700/80 flex items-center justify-center text-cyan-400 shrink-0">
                <FileAudio className="w-5 h-5" />
              </div>
              <div className="overflow-hidden">
                <p className="text-sm font-medium text-slate-200 truncate">{file.name}</p>
                <p className="text-xs text-slate-400 font-mono">
                  {(file.size / (1024 * 1024)).toFixed(2)} MB • {file.type || "audio"}
                </p>
              </div>
            </div>
            <button
              onClick={() => onFileSelect(null)}
              className="p-1.5 rounded-lg bg-slate-800/60 hover:bg-rose-950/60 text-slate-400 hover:text-rose-300 transition-colors"
              title="Remove file"
              aria-label="Remove uploaded file"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          <WaveformVisualizer file={file} label="Carrier Signal Waveform" variant={accent} />
        </div>
      )}
    </div>
  );
}

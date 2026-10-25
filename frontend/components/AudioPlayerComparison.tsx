"use client";

import React from "react";
import { Download, ShieldCheck, Sparkles, Volume2 } from "lucide-react";
import { WaveformVisualizer } from "./WaveformVisualizer";

interface AudioPlayerComparisonProps {
  originalFile: File;
  stegoUrl: string;
  snrDb: number;
  chunks: number;
  duration: number;
  onDownload: () => void;
}

export function AudioPlayerComparison({
  originalFile,
  stegoUrl,
  snrDb,
  chunks,
  duration,
  onDownload,
}: AudioPlayerComparisonProps) {
  const originalUrl = React.useMemo(() => {
    return URL.createObjectURL(originalFile);
  }, [originalFile]);

  React.useEffect(() => {
    return () => {
      URL.revokeObjectURL(originalUrl);
    };
  }, [originalUrl]);

  // Plain language interpretation of SNR
  const getSnrExplanation = (snr: number) => {
    if (snr >= 70) {
      return {
        level: "Mathematically Imperceptible",
        desc: "Audio quality exceeds studio master thresholds. The embedded neural residual is far below human auditory perception and standard playback floor noise.",
        badge: "bg-emerald-500/20 text-emerald-300 border-emerald-500/30",
      };
    } else if (snr >= 45) {
      return {
        level: "Near-Lossless Transparency",
        desc: "Imperceptible in standard listening conditions. Residual frequencies are harmonically masked.",
        badge: "bg-cyan-500/20 text-cyan-300 border-cyan-500/30",
      };
    } else {
      return {
        level: "Noticeable Masking",
        desc: "Subtle background texture might be audible on high-fidelity headphones.",
        badge: "bg-amber-500/20 text-amber-300 border-amber-500/30",
      };
    }
  };

  const snrInfo = getSnrExplanation(snrDb);

  return (
    <div className="space-y-6 pt-4 border-t border-slate-800 animate-fadeIn">
      {/* SNR Metric Card */}
      <div className="bg-[#0b1220] border border-emerald-500/30 rounded-xl p-5 shadow-lg relative overflow-hidden">
        <div className="absolute top-0 right-0 w-32 h-32 bg-emerald-500/5 rounded-full blur-2xl pointer-events-none" />
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <ShieldCheck className="w-5 h-5 text-emerald-400" />
              <h4 className="text-base font-semibold text-slate-100">
                Steganographic Perturbation Analysis
              </h4>
            </div>
            <p className="text-xs text-slate-400 mt-1 max-w-xl leading-relaxed">
              {snrInfo.desc}
            </p>
          </div>

          <div className="flex items-center gap-3">
            <div className="text-right font-mono">
              <span className="text-xs text-slate-400 uppercase tracking-widest block">Signal-to-Noise</span>
              <span className="text-2xl font-bold text-emerald-400">{snrDb.toFixed(1)} <span className="text-sm font-normal text-slate-400">dB</span></span>
            </div>
            <span className={`text-xs px-2.5 py-1.5 rounded-md border font-mono font-medium ${snrInfo.badge}`}>
              {snrInfo.level}
            </span>
          </div>
        </div>

        <div className="mt-4 pt-3 border-t border-slate-800/80 grid grid-cols-2 sm:grid-cols-3 gap-2 text-xs font-mono text-slate-400">
          <div>Chunks processed: <span className="text-slate-200 font-semibold">{chunks} ({(chunks * 2)}s capacity)</span></div>
          <div>Audio length: <span className="text-slate-200 font-semibold">{duration.toFixed(2)}s</span></div>
          <div>Sample format: <span className="text-emerald-400 font-semibold">16-bit PCM WAV</span></div>
        </div>
      </div>

      {/* Side-by-side Audio Comparison Players */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        {/* Original Audio Player */}
        <div className="bg-[#090e18] border border-slate-800 rounded-xl p-4 shadow-sm flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-mono uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                <Volume2 className="w-3.5 h-3.5 text-slate-400" />
                Original Cover Audio
              </span>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300">
                Unmodified
              </span>
            </div>
            <p className="text-sm font-medium text-slate-200 truncate mb-3">{originalFile.name}</p>
            <div className="mb-3">
              <WaveformVisualizer file={originalFile} label="Original Waveform" variant="cyan" />
            </div>
          </div>
          <audio controls src={originalUrl} className="w-full mt-2 filter invert hue-rotate-180 brightness-90 rounded" />
        </div>

        {/* Stego Audio Player */}
        <div className="bg-[#090e18] border border-emerald-500/30 rounded-xl p-4 shadow-md flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-mono uppercase tracking-wider text-emerald-400 flex items-center gap-1.5">
                <Sparkles className="w-3.5 h-3.5 text-emerald-400" />
                Stego Audio (Encoded)
              </span>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-950/60 border border-emerald-500/30 text-emerald-300">
                Secret Inside
              </span>
            </div>
            <p className="text-sm font-medium text-slate-200 truncate mb-3">stego.wav (16-bit PCM)</p>
            <div className="mb-3">
              <WaveformVisualizer file={null} audioUrl={stegoUrl} label="Stego Waveform" variant="emerald" />
            </div>
          </div>
          <audio controls src={stegoUrl} className="w-full mt-2 filter invert hue-rotate-180 brightness-90 rounded" />
        </div>
      </div>

      {/* Action Download Button */}
      <div className="flex justify-end pt-2">
        <button
          id="download-stego-button"
          onClick={onDownload}
          className="flex items-center gap-2.5 px-6 py-3 rounded-xl bg-gradient-to-r from-emerald-600 to-teal-500 hover:from-emerald-500 hover:to-teal-400 text-slate-950 font-semibold text-sm shadow-lg shadow-emerald-950/40 hover:shadow-emerald-900/60 transition-all focus-visible:ring-2 focus-visible:ring-emerald-400"
        >
          <Download className="w-4 h-4 text-slate-950" />
          <span>Download Stego Audio (stego.wav)</span>
        </button>
      </div>
    </div>
  );
}

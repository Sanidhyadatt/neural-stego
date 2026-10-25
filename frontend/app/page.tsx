"use client";

import React, { useEffect, useState } from "react";
import {
  Activity,
  Cpu,
  KeyRound,
  Lock,
  Radio,
  ShieldAlert,
  Sparkles,
} from "lucide-react";
import { HideTab } from "@/components/HideTab";
import { RevealTab } from "@/components/RevealTab";
import { checkHealth } from "@/lib/api";

export default function Home() {
  const [activeTab, setActiveTab] = useState<"hide" | "reveal">("hide");
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const [modelLoaded, setModelLoaded] = useState<boolean | null>(null);

  useEffect(() => {
    let isCancelled = false;

    async function checkBackend() {
      try {
        const health = await checkHealth();
        if (!isCancelled) {
          setBackendOnline(true);
          setModelLoaded(health.model_loaded);
        }
      } catch {
        if (!isCancelled) {
          setBackendOnline(false);
          setModelLoaded(false);
        }
      }
    }

    checkBackend();
    const interval = setInterval(checkBackend, 2000);
    return () => {
      isCancelled = true;
      clearInterval(interval);
    };
  }, []);

  return (
    <main className="min-h-screen py-8 px-4 sm:px-6 lg:px-8 max-w-4xl mx-auto flex flex-col justify-between">
      {/* Top Bar / Brand Header */}
      <header className="space-y-4 pb-6 border-b border-slate-800/80">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500/20 to-emerald-500/20 border border-cyan-500/40 flex items-center justify-center text-cyan-400 shadow-inner">
              <Activity className="w-6 h-6 animate-pulse" />
            </div>
            <div>
              <h1 className="text-xl sm:text-2xl font-bold tracking-tight text-slate-100 flex items-center gap-2">
                <span>Neural Audio Steganography</span>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-cyan-950 border border-cyan-500/30 text-cyan-400 font-semibold tracking-wider uppercase">
                  GAN STFT
                </span>
              </h1>
              <p className="text-xs text-slate-400 font-mono mt-0.5">
                Acoustic Concealment & Watermarking via Deep U-Net Residual Generator
              </p>
            </div>
          </div>

          {/* Backend Status Pill */}
          <div
            id="backend-status-pill"
            className="flex items-center gap-2 self-start sm:self-auto text-xs font-mono px-3 py-1.5 rounded-full bg-slate-900/90 border border-slate-800"
          >
            <span
              className={`w-2 h-2 rounded-full ${
                backendOnline && modelLoaded
                  ? "bg-emerald-400 shadow-sm shadow-emerald-400 animate-pulse"
                  : backendOnline && !modelLoaded
                  ? "bg-amber-400"
                  : "bg-rose-500"
              }`}
            />
            <span className="text-slate-300">
              {backendOnline && modelLoaded ? (
                "Neural Engine Online"
              ) : backendOnline && !modelLoaded ? (
                "Checkpoint Missing (503)"
              ) : (
                "Backend Disconnected"
              )}
            </span>
          </div>
        </div>

        {/* Warning if backend is offline */}
        {backendOnline === false && (
          <div
            id="backend-offline-banner"
            role="alert"
            className="flex items-center gap-2 text-xs font-mono p-3 rounded-xl bg-rose-950/40 border border-rose-500/40 text-rose-200"
          >
            <ShieldAlert className="w-4 h-4 text-rose-400 shrink-0" />
            <span>
              Backend server is currently disconnected (failed to connect to <code>http://localhost:8000</code>). Please ensure the FastAPI backend is running.
            </span>
          </div>
        )}

        {/* Warning if checkpoint is missing */}
        {backendOnline && !modelLoaded && (
          <div
            role="alert"
            className="flex items-center gap-2 text-xs font-mono p-3 rounded-xl bg-amber-950/30 border border-amber-500/30 text-amber-200"
          >
            <ShieldAlert className="w-4 h-4 text-amber-400 shrink-0" />
            <span>
              Model weights not detected at <code>checkpoints/stego.pt</code>. Train a model or place weights in the checkpoints directory.
            </span>
          </div>
        )}
      </header>

      {/* Main Steganography Interface Card */}
      <section className="my-8 bg-[#0a0f1c]/90 border border-slate-800/90 rounded-2xl p-4 sm:p-7 shadow-2xl backdrop-blur-xl oscilloscope-border">
        {/* Accessible Tab Navigation */}
        <div
          role="tablist"
          aria-label="Steganography Actions"
          className="flex p-1 mb-6 rounded-xl bg-[#060a12] border border-slate-800 grid grid-cols-2 gap-1 font-mono text-xs"
        >
          <button
            role="tab"
            id="tab-hide"
            aria-selected={activeTab === "hide"}
            aria-controls="panel-hide"
            tabIndex={activeTab === "hide" ? 0 : -1}
            onClick={() => setActiveTab("hide")}
            className={`flex items-center justify-center gap-2 py-2.5 px-4 rounded-lg font-semibold transition-all ${
              activeTab === "hide"
                ? "bg-gradient-to-r from-cyan-950 to-slate-900 border border-cyan-500/40 text-cyan-300 shadow-md shadow-cyan-950/50"
                : "text-slate-400 hover:text-slate-200 hover:bg-slate-900/40"
            }`}
          >
            <Lock className="w-3.5 h-3.5" />
            <span>Hide a message</span>
          </button>

          <button
            role="tab"
            id="tab-reveal"
            aria-selected={activeTab === "reveal"}
            aria-controls="panel-reveal"
            tabIndex={activeTab === "reveal" ? 0 : -1}
            onClick={() => setActiveTab("reveal")}
            className={`flex items-center justify-center gap-2 py-2.5 px-4 rounded-lg font-semibold transition-all ${
              activeTab === "reveal"
                ? "bg-gradient-to-r from-emerald-950 to-slate-900 border border-emerald-500/40 text-emerald-300 shadow-md shadow-emerald-950/50"
                : "text-slate-400 hover:text-slate-200 hover:bg-slate-900/40"
            }`}
          >
            <KeyRound className="w-3.5 h-3.5" />
            <span>Reveal a message</span>
          </button>
        </div>

        {/* Tab Panels */}
        <div
          role="tabpanel"
          id="panel-hide"
          aria-labelledby="tab-hide"
          hidden={activeTab !== "hide"}
        >
          {activeTab === "hide" && <HideTab />}
        </div>

        <div
          role="tabpanel"
          id="panel-reveal"
          aria-labelledby="tab-reveal"
          hidden={activeTab !== "reveal"}
        >
          {activeTab === "reveal" && <RevealTab />}
        </div>
      </section>

      {/* Footer System Specs */}
      <footer className="pt-6 border-t border-slate-800/80 text-xs font-mono text-slate-500 flex flex-col sm:flex-row items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-4">
          <span className="flex items-center gap-1.5">
            <Radio className="w-3.5 h-3.5 text-cyan-400/80" />
            16 kHz Mono PCM
          </span>
          <span className="flex items-center gap-1.5">
            <Cpu className="w-3.5 h-3.5 text-emerald-400/80" />
            128 bits / 2s Block
          </span>
          <span className="flex items-center gap-1.5">
            <Sparkles className="w-3.5 h-3.5 text-amber-400/80" />
            Reed-Solomon (8 Parity)
          </span>
        </div>
        <p className="text-[11px] text-slate-500">
          Neural Audio Steganography System
        </p>
      </footer>
    </main>
  );
}

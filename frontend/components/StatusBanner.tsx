"use client";

import React from "react";
import { AlertCircle, CheckCircle2, Info, X } from "lucide-react";

interface StatusBannerProps {
  type: "success" | "error" | "info" | "warning";
  message: string;
  details?: string;
  onClose?: () => void;
}

export function StatusBanner({ type, message, details, onClose }: StatusBannerProps) {
  if (!message) return null;

  const styles = {
    success: {
      bg: "bg-emerald-950/40 border-emerald-500/40 text-emerald-200",
      icon: <CheckCircle2 className="w-5 h-5 text-emerald-400 shrink-0" />,
    },
    error: {
      bg: "bg-rose-950/40 border-rose-500/40 text-rose-200",
      icon: <AlertCircle className="w-5 h-5 text-rose-400 shrink-0" />,
    },
    warning: {
      bg: "bg-amber-950/40 border-amber-500/40 text-amber-200",
      icon: <AlertCircle className="w-5 h-5 text-amber-400 shrink-0" />,
    },
    info: {
      bg: "bg-cyan-950/40 border-cyan-500/40 text-cyan-200",
      icon: <Info className="w-5 h-5 text-cyan-400 shrink-0" />,
    },
  }[type];

  return (
    <div
      role="alert"
      aria-live="polite"
      className={`w-full rounded-lg border p-4 backdrop-blur-md transition-all shadow-lg ${styles.bg}`}
    >
      <div className="flex items-start gap-3">
        {styles.icon}
        <div className="flex-1 text-sm">
          <p className="font-semibold leading-relaxed">{message}</p>
          {details && <p className="mt-1 text-xs opacity-80 leading-relaxed font-mono">{details}</p>}
        </div>
        {onClose && (
          <button
            onClick={onClose}
            className="p-1 rounded hover:bg-white/10 transition-colors focus:ring-1 focus:ring-slate-400"
            aria-label="Close notification"
          >
            <X className="w-4 h-4 opacity-70" />
          </button>
        )}
      </div>
    </div>
  );
}

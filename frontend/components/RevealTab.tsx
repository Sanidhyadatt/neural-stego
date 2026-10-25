"use client";

import React, { useState } from "react";
import { Check, Copy, KeyRound, Sparkles } from "lucide-react";
import { AudioUploader } from "./AudioUploader";
import { StatusBanner } from "./StatusBanner";
import { ApiError, extractMessage } from "@/lib/api";

export function RevealTab() {
  const [file, setFile] = useState<File | null>(null);
  const [isExtracting, setIsExtracting] = useState(false);
  const [extractedMessage, setExtractedMessage] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState<{ message: string; details?: string } | null>(null);

  const handleFileSelect = (selectedFile: File | null) => {
    setFile(selectedFile);
    setExtractedMessage(null);
    setError(null);
  };

  const handleExtract = async () => {
    if (!file) return;

    setIsExtracting(true);
    setExtractedMessage(null);
    setError(null);

    try {
      const res = await extractMessage(file);
      setExtractedMessage(res.message);
    } catch (err: unknown) {
      if (err instanceof ApiError) {
        setError({ message: err.message });
      } else {
        setError({
          message: "Failed to extract message",
          details: "An unexpected error occurred while communicating with the extraction API.",
        });
      }
    } finally {
      setIsExtracting(false);
    }
  };

  const handleCopy = () => {
    if (!extractedMessage) return;
    navigator.clipboard.writeText(extractedMessage);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="space-y-6">
      {/* Upload Zone */}
      <AudioUploader
        id="reveal-file-input"
        label="1. Upload Stego Audio File"
        file={file}
        onFileSelect={handleFileSelect}
        onError={(errMsg) => setError({ message: errMsg })}
        accent="emerald"
      />

      {/* Error or Alert banner */}
      {error && (
        <StatusBanner
          type="error"
          message={error.message}
          details={error.details}
          onClose={() => setError(null)}
        />
      )}

      {/* Action Button */}
      <button
        id="reveal-message-button"
        onClick={handleExtract}
        disabled={!file || isExtracting}
        className="w-full flex items-center justify-center gap-2 py-3.5 px-6 rounded-xl font-semibold text-sm transition-all focus-visible:ring-2 focus-visible:ring-emerald-400 shadow-lg disabled:opacity-40 disabled:cursor-not-allowed bg-gradient-to-r from-emerald-600 via-teal-600 to-cyan-600 hover:from-emerald-500 hover:via-teal-500 hover:to-cyan-500 text-slate-950 font-bold"
      >
        {isExtracting ? (
          <>
            <div className="w-4 h-4 border-2 border-slate-950 border-t-transparent rounded-full animate-spin" />
            <span>Extracting Secret Message via Neural Decoder...</span>
          </>
        ) : (
          <>
            <KeyRound className="w-4 h-4" />
            <span>Reveal Hidden Message</span>
          </>
        )}
      </button>

      {/* Recovered Secret Display */}
      {extractedMessage !== null && (
        <div id="revealed-message-container" className="bg-[#0b1322] border border-emerald-500/40 rounded-xl p-5 shadow-xl space-y-3 relative overflow-hidden animate-fadeIn">
          <div className="absolute top-0 right-0 w-32 h-32 bg-emerald-500/10 rounded-full blur-2xl pointer-events-none" />

          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-emerald-400" />
              <h4 className="text-sm font-semibold text-slate-100 font-mono">
                Decoded Secret Payload ({extractedMessage.length} chars)
              </h4>
            </div>

            <button
              onClick={handleCopy}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800/80 hover:bg-slate-700 text-xs font-mono text-slate-200 transition-colors"
              aria-label="Copy message to clipboard"
            >
              {copied ? (
                <>
                  <Check className="w-3.5 h-3.5 text-emerald-400" />
                  <span className="text-emerald-400">Copied!</span>
                </>
              ) : (
                <>
                  <Copy className="w-3.5 h-3.5 text-slate-400" />
                  <span>Copy</span>
                </>
              )}
            </button>
          </div>

          <div id="revealed-message-display" className="p-4 rounded-lg bg-[#060a12] border border-slate-800/80 text-emerald-300 font-mono text-sm leading-relaxed whitespace-pre-wrap select-all">
            {extractedMessage}
          </div>
        </div>
      )}
    </div>
  );
}

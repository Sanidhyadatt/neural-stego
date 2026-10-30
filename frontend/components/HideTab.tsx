"use client";

import React, { useMemo, useState } from "react";
import { Lock } from "lucide-react";
import { AudioPlayerComparison } from "./AudioPlayerComparison";
import { AudioUploader } from "./AudioUploader";
import { StatusBanner } from "./StatusBanner";
import { ApiError, EmbedResult, embedMessage, getCapacity } from "@/lib/api";

const MAX_PAYLOAD_BYTES = 255;

function computeNeededSeconds(payloadBytes: number): number {
  const frameLen = payloadBytes + 4; // 1 magic + 1 length + 2 CRC16
  const rsBlocks = Math.ceil(frameLen / 247);
  const encodedLen = frameLen + 8 * rsBlocks;
  const nBlocks = Math.ceil(encodedLen / 16);
  return nBlocks * 2;
}

export function HideTab() {
  const [file, setFile] = useState<File | null>(null);
  const [capacityChars, setCapacityChars] = useState<number | null>(null);
  const [durationSec, setDurationSec] = useState<number | null>(null);
  const [isQueryingCapacity, setIsQueryingCapacity] = useState(false);

  const [message, setMessage] = useState("");
  const [isEmbedding, setIsEmbedding] = useState(false);
  const [embedResult, setEmbedResult] = useState<EmbedResult | null>(null);

  const [error, setError] = useState<{ message: string; details?: string } | null>(null);
  const [progressStep, setProgressStep] = useState<string>("");

  const msgByteLength = useMemo(
    () => new TextEncoder().encode(message).length,
    [message]
  );

  const handleFileSelect = async (selectedFile: File | null) => {
    setFile(selectedFile);
    setCapacityChars(null);
    setDurationSec(null);
    setEmbedResult(null);
    setError(null);

    if (!selectedFile) return;

    setIsQueryingCapacity(true);
    try {
      const cap = await getCapacity(selectedFile);
      setCapacityChars(cap.max_chars);
      setDurationSec(cap.duration_sec);
    } catch (err: unknown) {
      if (err instanceof ApiError) {
        setError({ message: err.message });
      } else {
        setError({
          message: "Failed to read audio file capacity.",
          details: "Please verify server connectivity and supported audio formats.",
        });
      }
    } finally {
      setIsQueryingCapacity(false);
    }
  };

  const handleEmbed = async () => {
    if (!file || !message.trim()) return;

    setIsEmbedding(true);
    setError(null);
    setProgressStep("Transforming audio via STFT...");

    try {
      setTimeout(() => setProgressStep("Injecting 128-bit payload with U-Net Encoder..."), 600);
      setTimeout(() => setProgressStep("Synthesizing inverse STFT and computing SNR..."), 1200);

      const result = await embedMessage(file, message);
      setEmbedResult(result);
    } catch (err: unknown) {
      if (err instanceof ApiError) {
        setError({ message: err.message });
      } else {
        setError({
          message: "Embedding failed.",
          details: "An unexpected error occurred during audio encoding.",
        });
      }
    } finally {
      setIsEmbedding(false);
      setProgressStep("");
    }
  };

  const handleDownload = () => {
    if (!embedResult) return;
    const a = document.createElement("a");
    a.href = embedResult.url;
    a.download = embedResult.filename || "stego.wav";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  const isOverProtocolMax = msgByteLength > MAX_PAYLOAD_BYTES;
  const isOverCapacity =
    capacityChars !== null && msgByteLength > capacityChars;

  const capacityError = useMemo<{ message: string; details?: string } | null>(() => {
    if (!file || msgByteLength === 0) return null;
    if (isOverProtocolMax) {
      return {
        message: `Message UTF-8 byte length (${msgByteLength}) exceeds maximum of ${MAX_PAYLOAD_BYTES} bytes.`,
      };
    }
    if (isOverCapacity) {
      const neededSec = computeNeededSeconds(msgByteLength);
      return {
        message: `Audio too short for this message. Needs at least ${neededSec} seconds.`,
      };
    }
    return null;
  }, [file, msgByteLength, isOverProtocolMax, isOverCapacity]);

  const activeError = error || capacityError;

  const isButtonDisabled =
    !file ||
    message.trim().length === 0 ||
    isOverProtocolMax ||
    isOverCapacity ||
    isEmbedding ||
    isQueryingCapacity ||
    capacityChars === null;

  return (
    <div className="space-y-6">
      {/* Upload Zone */}
      <AudioUploader
        id="hide-file-input"
        label="1. Choose Cover Audio"
        file={file}
        onFileSelect={handleFileSelect}
        onError={(errMsg) => setError({ message: errMsg })}
        accent="cyan"
      />

      {/* Capacity Announcement */}
      {isQueryingCapacity && (
        <div className="flex items-center gap-2 text-xs font-mono text-cyan-400 p-2.5 rounded-lg bg-cyan-950/20 border border-cyan-500/20">
          <div className="w-3.5 h-3.5 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin" />
          <span>Analyzing carrier audio and calculating payload capacity...</span>
        </div>
      )}

      {capacityChars !== null && !isQueryingCapacity && (
        <div className="flex flex-col sm:flex-row sm:items-center justify-between p-3.5 rounded-xl bg-slate-900/60 border border-slate-800 text-xs font-mono text-slate-300">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            <span>Carrier duration: <strong className="text-slate-100">{durationSec}s</strong></span>
          </div>
          <div className="mt-1 sm:mt-0">
            Available payload:{" "}
            <span className="text-emerald-400 font-bold text-sm">
              Fits up to {capacityChars} UTF-8 bytes
            </span>
          </div>
        </div>
      )}

      {/* Secret Message Input */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <label htmlFor="secret-message" className="text-xs font-mono font-medium text-slate-300 uppercase tracking-wider">
            2. Enter Secret Message
          </label>
          {capacityChars !== null && (
            <span
              className={`text-xs font-mono font-semibold ${
                isOverCapacity || isOverProtocolMax
                  ? "text-rose-400"
                  : msgByteLength > capacityChars * 0.8
                  ? "text-amber-400"
                  : "text-slate-400"
              }`}
            >
              {msgByteLength} / {capacityChars} bytes
            </span>
          )}
        </div>

        <textarea
          id="secret-message"
          rows={3}
          value={message}
          onChange={(e) => {
            setMessage(e.target.value);
            if (error) setError(null);
          }}
          placeholder={
            file
              ? "Type your confidential message here..."
              : "Upload an audio file above first..."
          }
          disabled={!file || isEmbedding}
          className="w-full rounded-xl bg-[#080d16] border border-slate-800 p-3.5 text-sm text-slate-100 placeholder-slate-500 focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 transition-all font-mono disabled:opacity-50 disabled:cursor-not-allowed resize-none"
        />

        {/* Capacity Bar */}
        {capacityChars !== null && (
          <div className="w-full h-1.5 rounded-full bg-slate-900 overflow-hidden">
            <div
              className={`h-full transition-all duration-300 ${
                isOverCapacity || isOverProtocolMax
                  ? "bg-rose-500"
                  : msgByteLength > capacityChars * 0.8
                  ? "bg-amber-400"
                  : "bg-cyan-400"
              }`}
              style={{
                width: `${Math.min(100, (msgByteLength / (capacityChars || 1)) * 100)}%`,
              }}
            />
          </div>
        )}
      </div>

      {/* Persistent aria-live error notifications (including audio too short) */}
      <div aria-live="assertive" aria-atomic="true">
        {activeError && (
          <StatusBanner
            id="hide-error-banner"
            type="error"
            message={activeError.message}
            details={activeError.details}
            onClose={error ? () => setError(null) : undefined}
          />
        )}
      </div>

      {/* Action Button & Loading Indicator */}
      <div className="pt-2">
        <button
          id="hide-message-button"
          onClick={handleEmbed}
          disabled={isButtonDisabled}
          className="w-full flex items-center justify-center gap-2 py-3.5 px-6 rounded-xl font-semibold text-sm transition-all focus-visible:ring-2 focus-visible:ring-cyan-400 shadow-lg disabled:opacity-40 disabled:cursor-not-allowed bg-gradient-to-r from-cyan-600 via-teal-600 to-emerald-600 hover:from-cyan-500 hover:via-teal-500 hover:to-emerald-500 text-slate-950 font-bold"
        >
          {isEmbedding ? (
            <>
              <div className="w-4 h-4 border-2 border-slate-950 border-t-transparent rounded-full animate-spin" />
              <span>{progressStep || "Encoding Audio..."}</span>
            </>
          ) : (
            <>
              <Lock className="w-4 h-4" />
              <span>Hide Message in Audio</span>
            </>
          )}
        </button>
      </div>

      {/* Side-by-side Results Comparison */}
      {embedResult && file && (
        <AudioPlayerComparison
          originalFile={file}
          stegoUrl={embedResult.url}
          snrDb={embedResult.snrDb}
          chunks={embedResult.chunks}
          duration={embedResult.duration}
          onDownload={handleDownload}
        />
      )}
    </div>
  );
}

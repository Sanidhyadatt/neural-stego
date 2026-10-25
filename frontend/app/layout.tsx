import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Neural Audio Steganography | GAN Audio Concealment System",
  description:
    "Hide and reveal secret text inside audio signals imperceptibly using deep generative adversarial networks (GAN) and Reed-Solomon error correction.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="antialiased min-h-screen text-slate-100 selection:bg-cyan-500/30 selection:text-cyan-200">
        {children}
      </body>
    </html>
  );
}

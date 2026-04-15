import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Vibeframe",
  description: "Text prompt to design workflow for Vibeframe.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
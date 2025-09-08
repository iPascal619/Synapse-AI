import type { Metadata } from 'next'
import { Montserrat } from 'next/font/google'
import { GeistMono } from 'geist/font/mono'
import { Analytics } from '@vercel/analytics/next'
import './globals.css'

const montserrat = Montserrat({
  subsets: ['latin'],
  variable: '--font-sans',
  display: 'swap',
})

export const metadata: Metadata = {
  title: 'v0 App',
  description: 'Created with v0',
  generator: 'v0.app',
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" className={`${montserrat.variable}`}>
      <head>
        <style>{`
html {
  font-family: ${montserrat.style.fontFamily};
  --font-sans: ${montserrat.variable};
  --font-mono: ${GeistMono.variable};
}
        `}</style>
      </head>
      <body className={montserrat.className}>
        {children}
        <Analytics />
      </body>
    </html>
  )
}

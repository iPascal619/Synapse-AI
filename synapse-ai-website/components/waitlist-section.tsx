"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"

export function WaitlistSection() {
  const [email, setEmail] = useState("")
  const [isSubmitted, setIsSubmitted] = useState(false)
  const [isLoading, setIsLoading] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsLoading(true)
    
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1000))
    
    setIsSubmitted(true)
    setIsLoading(false)
    setEmail("")
  }

  return (
    <section className="w-full py-20 px-5 relative overflow-hidden">
      {/* Background Elements */}
      <div className="absolute inset-0">
        <div className="w-[600px] h-[600px] absolute top-[-200px] left-1/2 -translate-x-1/2 bg-primary/5 rounded-full blur-3xl"></div>
        <div className="w-[400px] h-[400px] absolute bottom-[-100px] right-[-100px] bg-primary/3 rounded-full blur-2xl"></div>
      </div>
      
      <div className="max-w-4xl mx-auto relative z-10">
        <div className="text-center mb-12">
          {/* Status Badge */}
          <div className="inline-flex items-center px-4 py-2 rounded-full bg-amber-100 text-amber-800 text-sm font-medium mb-6 border border-amber-200">
            <div className="w-2 h-2 bg-amber-500 rounded-full mr-2 animate-pulse"></div>
            Currently in Development
          </div>
          
          <h2 className="text-4xl md:text-5xl lg:text-6xl font-bold text-foreground mb-6">
            Be the First to Experience
            <span className="text-primary block">Next-Gen Fraud Protection</span>
          </h2>
          
          <p className="text-lg md:text-xl text-muted-foreground max-w-3xl mx-auto mb-8">
            Our AI-powered fraud detection system is being fine-tuned to perfection. Join our exclusive waitlist to get early access and special launch pricing.
          </p>
        </div>

        {/* Waitlist Form */}
        <div className="max-w-md mx-auto">
          {!isSubmitted ? (
            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="flex flex-col sm:flex-row gap-3">
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="Enter your email address"
                  required
                  className="flex-1 px-4 py-3 rounded-lg border border-border bg-background text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
                />
                <Button
                  type="submit"
                  disabled={isLoading}
                  className="px-8 py-3 bg-primary text-primary-foreground hover:bg-primary/90 rounded-lg font-semibold transition-all duration-200 disabled:opacity-50"
                >
                  {isLoading ? (
                    <div className="flex items-center gap-2">
                      <div className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin"></div>
                      Joining...
                    </div>
                  ) : (
                    "Join Waitlist"
                  )}
                </Button>
              </div>
              <p className="text-xs text-muted-foreground text-center">
                No spam, unsubscribe at any time. We respect your privacy.
              </p>
            </form>
          ) : (
            <div className="text-center py-8">
              <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <svg className="w-8 h-8 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold text-foreground mb-2">You're on the list!</h3>
              <p className="text-muted-foreground">
                We'll notify you as soon as Synapse AI is ready for early access.
              </p>
            </div>
          )}
        </div>

        {/* Benefits */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-16">
          <div className="text-center">
            <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center mx-auto mb-4">
              <svg className="w-6 h-6 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            </div>
            <h3 className="font-semibold text-foreground mb-2">Early Access</h3>
            <p className="text-sm text-muted-foreground">
              Be among the first to test our revolutionary fraud detection system
            </p>
          </div>
          
          <div className="text-center">
            <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center mx-auto mb-4">
              <svg className="w-6 h-6 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1" />
              </svg>
            </div>
            <h3 className="font-semibold text-foreground mb-2">Special Pricing</h3>
            <p className="text-sm text-muted-foreground">
              Exclusive launch discount for early adopters and beta testers
            </p>
          </div>
          
          <div className="text-center">
            <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center mx-auto mb-4">
              <svg className="w-6 h-6 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
              </svg>
            </div>
            <h3 className="font-semibold text-foreground mb-2">Shape the Future</h3>
            <p className="text-sm text-muted-foreground">
              Your feedback will help us build the perfect fraud protection solution
            </p>
          </div>
        </div>

        {/* Social Proof */}
        <div className="text-center mt-12 pt-8 border-t border-border">
          <p className="text-sm text-muted-foreground mb-4">Already trusted by leading companies</p>
          <div className="flex flex-wrap justify-center items-center gap-8 opacity-60">
            <span className="text-lg font-semibold">Sony</span>
            <span className="text-lg font-semibold">McDonald's</span>
            <span className="text-lg font-semibold">IBM</span>
            <span className="text-lg font-semibold">MasterCard</span>
            <span className="text-lg font-semibold">Apple</span>
          </div>
        </div>
      </div>
    </section>
  )
}

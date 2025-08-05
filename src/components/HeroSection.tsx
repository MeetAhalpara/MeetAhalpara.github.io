import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { ArrowDown, Brain, Zap, Code2 } from "lucide-react";

export const HeroSection = () => {
  const [currentText, setCurrentText] = useState("");
  const [currentIndex, setCurrentIndex] = useState(0);
  const texts = [
    "Build AI Models",
    "Create AI Agents", 
    "Deploy to Production",
    "Scale with MLOps"
  ];

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentIndex((prev) => (prev + 1) % texts.length);
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const text = texts[currentIndex];
    let index = 0;
    const typeInterval = setInterval(() => {
      setCurrentText(text.slice(0, index));
      index++;
      if (index > text.length) {
        clearInterval(typeInterval);
      }
    }, 100);
    return () => clearInterval(typeInterval);
  }, [currentIndex]);

  const scrollToContent = () => {
    document.getElementById('ai-guide')?.scrollIntoView({ behavior: 'smooth' });
  };

  return (
    <section className="min-h-screen flex items-center justify-center relative overflow-hidden">
      {/* Animated Background */}
      <div className="absolute inset-0">
        <div className="absolute inset-0 bg-gradient-to-br from-background via-background to-muted/20" />
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-ai-primary/20 rounded-full blur-3xl animate-float" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-ai-secondary/20 rounded-full blur-3xl animate-float" style={{ animationDelay: '2s' }} />
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-64 h-64 bg-ai-neural/20 rounded-full blur-3xl animate-float" style={{ animationDelay: '4s' }} />
      </div>

      {/* Content */}
      <div className="relative z-10 text-center px-4 max-w-6xl mx-auto">
        <div className="mb-8 animate-slide-up">
          <div className="flex items-center justify-center gap-4 mb-4">
            <Brain className="w-12 h-12 text-ai-primary animate-pulse-glow" />
            <Zap className="w-10 h-10 text-ai-neural animate-pulse-glow" style={{ animationDelay: '0.5s' }} />
            <Code2 className="w-12 h-12 text-ai-accent animate-pulse-glow" style={{ animationDelay: '1s' }} />
          </div>
        </div>

        <h1 className="text-6xl md:text-8xl font-bold mb-6 animate-slide-up">
          <span className="bg-gradient-to-r from-ai-primary via-ai-secondary to-ai-neural bg-clip-text text-transparent animate-gradient-x">
            Learn to
          </span>
        </h1>
        
        <div className="h-24 mb-8 animate-slide-up" style={{ animationDelay: '0.3s' }}>
          <h2 className="text-4xl md:text-6xl font-bold bg-gradient-to-r from-ai-accent via-ai-code to-ai-primary bg-clip-text text-transparent min-h-[1.2em] animate-gradient-x">
            {currentText}
            <span className="animate-pulse">|</span>
          </h2>
        </div>

        <p className="text-xl md:text-2xl text-muted-foreground mb-12 max-w-3xl mx-auto animate-slide-up" style={{ animationDelay: '0.6s' }}>
          Master AI development from scratch with step-by-step Python tutorials, 
          deployment strategies, and production-ready code examples.
        </p>

        <div className="flex flex-col sm:flex-row gap-6 justify-center items-center animate-slide-up" style={{ animationDelay: '0.9s' }}>
          <Button 
            onClick={scrollToContent}
            size="lg" 
            className="text-lg px-8 py-6 rounded-full gradient-primary hover:scale-105 transition-smooth glow-ai group"
          >
            Start Your AI Journey
            <ArrowDown className="ml-2 w-5 h-5 group-hover:translate-y-1 transition-smooth" />
          </Button>
          
          <div className="flex items-center gap-4 text-sm text-muted-foreground">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-ai-primary animate-pulse" />
              <span>Python Code Examples</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-ai-neural animate-pulse" style={{ animationDelay: '0.5s' }} />
              <span>Production Ready</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-ai-accent animate-pulse" style={{ animationDelay: '1s' }} />
              <span>Step-by-Step Guide</span>
            </div>
          </div>
        </div>

        {/* Floating Elements */}
        <div className="absolute top-20 left-10 w-16 h-16 border border-ai-primary/30 rounded-lg animate-float backdrop-blur-glass" />
        <div className="absolute bottom-20 right-10 w-12 h-12 border border-ai-secondary/30 rounded-full animate-float backdrop-blur-glass" style={{ animationDelay: '1s' }} />
        <div className="absolute top-1/2 right-20 w-8 h-8 border border-ai-neural/30 rounded-lg animate-float backdrop-blur-glass" style={{ animationDelay: '2s' }} />
      </div>
    </section>
  );
};
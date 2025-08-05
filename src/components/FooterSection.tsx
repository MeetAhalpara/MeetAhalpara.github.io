import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Github, BookOpen, Zap, Heart, ArrowUp } from "lucide-react";

export const FooterSection = () => {
  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <footer className="py-20 px-4 border-t border-muted/20">
      <div className="max-w-7xl mx-auto">
        {/* Call to Action */}
        <Card className="p-12 mb-16 text-center gradient-primary backdrop-blur-glass border-0 shadow-neural animate-slide-up">
          <div className="max-w-3xl mx-auto">
            <h3 className="text-3xl md:text-4xl font-bold text-white mb-6">
              Ready to Build Your AI Future?
            </h3>
            <p className="text-xl text-white/90 mb-8 leading-relaxed">
              Start implementing these techniques today and join thousands of developers 
              creating the next generation of AI applications.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Button 
                size="lg" 
                variant="secondary"
                className="text-lg px-8 py-6 rounded-full bg-white/10 hover:bg-white/20 text-white border-white/20 backdrop-blur-sm transition-smooth hover:scale-105"
              >
                <Github className="mr-2 w-5 h-5" />
                View on GitHub
              </Button>
              <Button 
                size="lg" 
                variant="secondary"
                className="text-lg px-8 py-6 rounded-full bg-white/10 hover:bg-white/20 text-white border-white/20 backdrop-blur-sm transition-smooth hover:scale-105"
              >
                <BookOpen className="mr-2 w-5 h-5" />
                Documentation
              </Button>
            </div>
          </div>
        </Card>

        {/* Resources Grid */}
        <div className="grid md:grid-cols-3 gap-8 mb-16 animate-slide-up">
          <Card className="p-6 backdrop-blur-glass border-muted/20 hover:shadow-neural transition-smooth">
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2 rounded-lg bg-ai-primary/10 text-ai-primary">
                <Zap className="w-5 h-5" />
              </div>
              <h4 className="text-lg font-semibold">Quick Start</h4>
            </div>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li>• Set up development environment</li>
              <li>• Install required dependencies</li>
              <li>• Run your first AI model</li>
              <li>• Deploy to production</li>
            </ul>
          </Card>

          <Card className="p-6 backdrop-blur-glass border-muted/20 hover:shadow-neural transition-smooth">
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2 rounded-lg bg-ai-neural/10 text-ai-neural">
                <BookOpen className="w-5 h-5" />
              </div>
              <h4 className="text-lg font-semibold">Learning Path</h4>
            </div>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li>• Machine Learning Fundamentals</li>
              <li>• Deep Learning with TensorFlow</li>
              <li>• AI Agents with LangChain</li>
              <li>• Production MLOps</li>
            </ul>
          </Card>

          <Card className="p-6 backdrop-blur-glass border-muted/20 hover:shadow-neural transition-smooth">
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2 rounded-lg bg-ai-accent/10 text-ai-accent">
                <Heart className="w-5 h-5" />
              </div>
              <h4 className="text-lg font-semibold">Community</h4>
            </div>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li>• Join Discord discussions</li>
              <li>• Share your projects</li>
              <li>• Get help from experts</li>
              <li>• Contribute to open source</li>
            </ul>
          </Card>
        </div>

        {/* Bottom Section */}
        <div className="flex flex-col md:flex-row justify-between items-center gap-6 pt-8 border-t border-muted/20 animate-slide-up">
          <div className="text-center md:text-left">
            <p className="text-muted-foreground">
              Built with 💚 for the AI community • 
              <span className="text-ai-primary font-semibold"> Open Source & Free</span>
            </p>
            <p className="text-sm text-muted-foreground mt-1">
              Last updated: December 2024 • Python 3.9+ • TensorFlow 2.8+
            </p>
          </div>
          
          <Button 
            onClick={scrollToTop}
            variant="outline" 
            size="sm"
            className="rounded-full backdrop-blur-sm hover:scale-105 transition-smooth"
          >
            <ArrowUp className="w-4 h-4 mr-2" />
            Back to Top
          </Button>
        </div>
      </div>
    </footer>
  );
};
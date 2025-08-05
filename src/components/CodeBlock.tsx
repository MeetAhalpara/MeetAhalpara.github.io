import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Copy, Check } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

interface CodeBlockProps {
  code: string;
  language?: string;
  title?: string;
  description?: string;
}

export const CodeBlock = ({ code, language = "python", title, description }: CodeBlockProps) => {
  const [copied, setCopied] = useState(false);
  const { toast } = useToast();

  const copyToClipboard = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      toast({
        title: "Code copied!",
        description: "The code has been copied to your clipboard.",
        duration: 2000,
      });
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      toast({
        title: "Failed to copy",
        description: "Please try again.",
        variant: "destructive",
        duration: 2000,
      });
    }
  };

  const formatCode = (code: string) => {
    const lines = code.split('\n');
    return lines.map((line, index) => {
      // Simple syntax highlighting for Python
      let formattedLine = line;
      
      // Keywords
      formattedLine = formattedLine.replace(
        /\b(def|class|import|from|if|else|elif|for|while|try|except|finally|with|as|return|yield|lambda|and|or|not|in|is|True|False|None)\b/g, 
        '<span class="text-ai-secondary font-semibold">$1</span>'
      );
      
      // Strings
      formattedLine = formattedLine.replace(
        /(["'])((?:\\.|(?!\1)[^\\])*?)\1/g,
        '<span class="text-ai-accent">$1$2$1</span>'
      );
      
      // Comments
      formattedLine = formattedLine.replace(
        /(#.*$)/g,
        '<span class="text-muted-foreground italic">$1</span>'
      );
      
      // Numbers
      formattedLine = formattedLine.replace(
        /\b(\d+\.?\d*)\b/g,
        '<span class="text-ai-code">$1</span>'
      );
      
      // Functions
      formattedLine = formattedLine.replace(
        /\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(/g,
        '<span class="text-ai-neural">$1</span>('
      );

      return (
        <div key={index} className="flex">
          <span className="text-muted-foreground text-sm mr-4 select-none min-w-[2rem] text-right">
            {index + 1}
          </span>
          <span dangerouslySetInnerHTML={{ __html: formattedLine || '&nbsp;' }} />
        </div>
      );
    });
  };

  return (
    <div className="w-full bg-card border rounded-lg overflow-hidden shadow-neural animate-slide-up">
      {(title || description) && (
        <div className="px-6 py-4 border-b bg-muted/30">
          {title && (
            <h3 className="text-lg font-semibold text-foreground mb-1">{title}</h3>
          )}
          {description && (
            <p className="text-sm text-muted-foreground">{description}</p>
          )}
        </div>
      )}
      
      <div className="relative">
        <div className="absolute top-4 right-4 z-10">
          <Button
            variant="outline"
            size="sm"
            onClick={copyToClipboard}
            className="backdrop-blur-sm bg-background/80 hover:bg-background/90 transition-smooth"
          >
            {copied ? (
              <Check className="w-4 h-4 text-ai-primary" />
            ) : (
              <Copy className="w-4 h-4" />
            )}
          </Button>
        </div>
        
        <div className="p-6 pt-12 overflow-x-auto">
          <code className="text-sm font-mono leading-relaxed block">
            {formatCode(code)}
          </code>
        </div>
      </div>
    </div>
  );
};
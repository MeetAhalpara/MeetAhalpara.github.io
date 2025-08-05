import { HeroSection } from "@/components/HeroSection";
import { GuideSection } from "@/components/GuideSection";
import { FooterSection } from "@/components/FooterSection";

const Index = () => {
  return (
    <div className="min-h-screen bg-background">
      <HeroSection />
      <GuideSection />
      <FooterSection />
    </div>
  );
};

export default Index;

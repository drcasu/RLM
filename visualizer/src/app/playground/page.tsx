'use client';

import { ThemeToggle } from '@/components/ThemeToggle';
import { PlaygroundForm } from '@/components/PlaygroundForm';
import { PlaygroundResults } from '@/components/PlaygroundResults';
import { useState } from 'react';
import Link from 'next/link';

interface RunResult {
  success: boolean;
  response: string | null;
  root_model: string | null;
  execution_time: number | null;
  usage_summary: Record<string, any> | null;
  error: string | null;
}

export default function PlaygroundPage() {
  const [result, setResult] = useState<RunResult | null>(null);
  const [loading, setLoading] = useState(false);

  const handleRun = async (config: any) => {
    setLoading(true);
    setResult(null);

    try {
      const apiUrl = process.env.NEXT_PUBLIC_PLAYGROUND_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/api/run`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(config),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (error) {
      setResult({
        success: false,
        response: null,
        root_model: null,
        execution_time: null,
        usage_summary: null,
        error: error instanceof Error ? error.message : 'Unknown error occurred',
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background relative overflow-hidden">
      {/* Background effects */}
      <div className="absolute inset-0 grid-pattern opacity-30 dark:opacity-15" />
      <div className="absolute top-0 left-1/3 w-[500px] h-[500px] bg-primary/5 rounded-full blur-3xl" />
      <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-primary/3 rounded-full blur-3xl" />
      
      <div className="relative z-10">
        {/* Header */}
        <header className="border-b border-border">
          <div className="max-w-7xl mx-auto px-6 py-6">
            <div className="flex items-center justify-between">
              <div>
                <Link href="/">
                  <h1 className="text-3xl font-bold tracking-tight">
                    <span className="text-primary">RLM</span>
                    <span className="text-muted-foreground ml-2 font-normal">Playground</span>
                  </h1>
                </Link>
                <p className="text-sm text-muted-foreground mt-1">
                  Run recursive language model completions interactively
                </p>
              </div>
              <div className="flex items-center gap-4">
                <ThemeToggle />
                <div className="flex items-center gap-2 text-[10px] text-muted-foreground font-mono">
                  <span className="flex items-center gap-1.5">
                    <span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse" />
                    READY
                  </span>
                </div>
              </div>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <main className="max-w-7xl mx-auto px-6 py-8">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-10">
            <div className="space-y-4">
              <h2 className="text-sm font-medium mb-3 flex items-center gap-2 text-muted-foreground">
                <span className="text-primary font-mono">01</span>
                Configuration
              </h2>
              <PlaygroundForm onRun={handleRun} loading={loading} />
            </div>

            <div className="space-y-4">
              <h2 className="text-sm font-medium mb-3 flex items-center gap-2 text-muted-foreground">
                <span className="text-primary font-mono">02</span>
                Execution Results
              </h2>
              <PlaygroundResults result={result} loading={loading} />
            </div>
          </div>
        </main>

        {/* Footer */}
        <footer className="border-t border-border mt-8">
          <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
            <p className="text-[10px] text-muted-foreground font-mono">
              RLM Playground • Recursive Language Models
            </p>
            <p className="text-[10px] text-muted-foreground font-mono">
              [LM ↔ REPL]
            </p>
          </div>
        </footer>
      </div>
    </div>
  );
}


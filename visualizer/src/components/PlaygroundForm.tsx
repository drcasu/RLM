'use client';

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { KeyValueEditor } from './KeyValueEditor';

interface PlaygroundFormProps {
  onRun: (config: any) => void;
  loading: boolean;
}

interface KeyValuePair {
  key: string;
  value: string;
}

export function PlaygroundForm({ onRun, loading }: PlaygroundFormProps) {
  const [backend, setBackend] = useState<string>('openai');
  const [backendKwargs, setBackendKwargs] = useState<KeyValuePair[]>([
    { key: 'model_name', value: 'openai/gpt-4.1-mini' },
  ]);
  const [environment, setEnvironment] = useState<string>('local');
  const [environmentKwargs, setEnvironmentKwargs] = useState<KeyValuePair[]>([]);
  const [prompt, setPrompt] = useState<string>('Print me the first 10 Fibonacci numbers.');
  const [rootPrompt, setRootPrompt] = useState<string>('');
  const [maxIterations, setMaxIterations] = useState<number>(30);
  const [maxDepth, setMaxDepth] = useState<number>(1);
  const [customSystemPrompt, setCustomSystemPrompt] = useState<string>('');
  const [otherBackends, setOtherBackends] = useState<string[]>([]);
  const [otherBackendKwargs, setOtherBackendKwargs] = useState<KeyValuePair[][]>([]);
  const [verbose, setVerbose] = useState<boolean>(false);
  const [enableLogging, setEnableLogging] = useState<boolean>(false);

  // Convert key-value pairs to object
  const pairsToObject = (pairs: KeyValuePair[]): Record<string, any> => {
    const obj: Record<string, any> = {};
    pairs.forEach(({ key, value }) => {
      if (key.trim()) {
        // Try to parse as number or boolean, otherwise keep as string
        let parsedValue: any = value;
        if (value === 'true') parsedValue = true;
        else if (value === 'false') parsedValue = false;
        else if (value === 'null' || value === '') parsedValue = null;
        else if (!isNaN(Number(value)) && value.trim() !== '') {
          parsedValue = Number(value);
        }
        obj[key.trim()] = parsedValue;
      }
    });
    return obj;
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    const config: any = {
      prompt,
      root_prompt: rootPrompt || undefined,
      backend,
      backend_kwargs: pairsToObject(backendKwargs),
      environment,
      environment_kwargs: pairsToObject(environmentKwargs),
      max_iterations: maxIterations,
      max_depth: maxDepth,
      verbose: verbose,
      enable_logging: enableLogging,
    };

    if (customSystemPrompt.trim()) {
      config.custom_system_prompt = customSystemPrompt;
    }

    if (otherBackends.length > 0) {
      config.other_backends = otherBackends;
      if (otherBackendKwargs.length > 0) {
        config.other_backend_kwargs = otherBackendKwargs.map(pairsToObject);
      }
    }

    onRun(config);
  };

  const addOtherBackend = () => {
    setOtherBackends([...otherBackends, 'openai']);
    setOtherBackendKwargs([...otherBackendKwargs, []]);
  };

  const removeOtherBackend = (index: number) => {
    setOtherBackends(otherBackends.filter((_, i) => i !== index));
    setOtherBackendKwargs(otherBackendKwargs.filter((_, i) => i !== index));
  };

  const updateOtherBackend = (index: number, backend: string) => {
    const newBackends = [...otherBackends];
    newBackends[index] = backend;
    setOtherBackends(newBackends);
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Configuration</CardTitle>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Backend Selection */}
          <div className="space-y-2">
            <label htmlFor="backend" className="text-sm font-medium">
              Backend
            </label>
            <select
              id="backend"
              value={backend}
              onChange={(e) => setBackend(e.target.value)}
              className="w-full px-3 py-2 bg-background border border-input rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-ring"
              disabled={loading}
            >
              <option value="openai">OpenAI</option>
              <option value="anthropic">Anthropic</option>
              <option value="portkey">Portkey</option>
              <option value="openrouter">OpenRouter</option>
              <option value="vllm">vLLM</option>
              <option value="litellm">LiteLLM</option>
            </select>
          </div>

          {/* Backend Kwargs */}
          <div className="space-y-2">
            <label className="text-sm font-medium">Backend Kwargs</label>
            <p className="text-xs text-muted-foreground">
              Key-value pairs for backend configuration (e.g., model_name, api_key, base_url)
            </p>
            <KeyValueEditor
              pairs={backendKwargs}
              onChange={setBackendKwargs}
              disabled={loading}
              keyPlaceholder="e.g., model_name"
              valuePlaceholder="e.g., gpt-5-nano"
            />
          </div>

          {/* Environment */}
          <div className="space-y-2">
            <label htmlFor="environment" className="text-sm font-medium">
              Environment
            </label>
            <select
              id="environment"
              value={environment}
              onChange={(e) => setEnvironment(e.target.value)}
              className="w-full px-3 py-2 bg-background border border-input rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-ring"
              disabled={loading}
            >
              <option value="local">Local</option>
              <option value="modal">Modal</option>
              <option value="prime">Prime</option>
            </select>
          </div>

          {/* Environment Kwargs */}
          <div className="space-y-2">
            <label className="text-sm font-medium">Environment Kwargs</label>
            <p className="text-xs text-muted-foreground">
              Key-value pairs for environment configuration
            </p>
            <KeyValueEditor
              pairs={environmentKwargs}
              onChange={setEnvironmentKwargs}
              disabled={loading}
            />
          </div>

          {/* Prompt */}
          <div className="space-y-2">
            <label htmlFor="prompt" className="text-sm font-medium">
              Prompt
            </label>
            <textarea
              id="prompt"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              rows={6}
              className="w-full px-3 py-2 bg-background border border-input rounded-md text-sm font-mono focus:outline-none focus:ring-2 focus:ring-ring resize-y"
              placeholder="Enter your prompt here..."
              disabled={loading}
              required
            />
          </div>

          {/* Root Prompt */}
          <div className="space-y-2">
            <label htmlFor="rootPrompt" className="text-sm font-medium">
              Root Prompt (optional)
            </label>
            <input
              id="rootPrompt"
              type="text"
              value={rootPrompt}
              onChange={(e) => setRootPrompt(e.target.value)}
              className="w-full px-3 py-2 bg-background border border-input rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-ring"
              placeholder="Optional hint shown to the root LM"
              disabled={loading}
            />
          </div>

          {/* Custom System Prompt */}
          <div className="space-y-2">
            <label htmlFor="customSystemPrompt" className="text-sm font-medium">
              Custom System Prompt (optional)
            </label>
            <textarea
              id="customSystemPrompt"
              value={customSystemPrompt}
              onChange={(e) => setCustomSystemPrompt(e.target.value)}
              rows={4}
              className="w-full px-3 py-2 bg-background border border-input rounded-md text-sm font-mono focus:outline-none focus:ring-2 focus:ring-ring resize-y"
              placeholder="Override the default system prompt"
              disabled={loading}
            />
          </div>

          {/* Max Iterations */}
          <div className="space-y-2">
            <label htmlFor="maxIterations" className="text-sm font-medium">
              Max Iterations
            </label>
            <input
              id="maxIterations"
              type="number"
              value={maxIterations}
              onChange={(e) => setMaxIterations(parseInt(e.target.value) || 30)}
              min={1}
              max={100}
              className="w-full px-3 py-2 bg-background border border-input rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-ring"
              disabled={loading}
            />
          </div>

          {/* Max Depth */}
          <div className="space-y-2">
            <label htmlFor="maxDepth" className="text-sm font-medium">
              Max Depth
            </label>
            <select
              id="maxDepth"
              value={maxDepth}
              onChange={(e) => setMaxDepth(parseInt(e.target.value))}
              className="w-full px-3 py-2 bg-background border border-input rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-ring"
              disabled={loading}
            >
              <option value={0}>0 (No recursion)</option>
              <option value={1}>1 (One level of recursion)</option>
            </select>
          </div>

          {/* Other Backends */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="text-sm font-medium">Other Backends (for sub-calls)</label>
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={addOtherBackend}
                disabled={loading}
              >
                + Add Backend
              </Button>
            </div>
            <p className="text-xs text-muted-foreground">
              Additional backends that environments can use for sub-LM calls
            </p>
            {otherBackends.map((otherBackend, index) => (
              <div key={index} className="space-y-2 p-3 border border-input rounded-md">
                <div className="flex items-center gap-2">
                  <select
                    value={otherBackend}
                    onChange={(e) => updateOtherBackend(index, e.target.value)}
                    className="flex-1 px-3 py-2 bg-background border border-input rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                    disabled={loading}
                  >
                    <option value="openai">OpenAI</option>
                    <option value="anthropic">Anthropic</option>
                    <option value="portkey">Portkey</option>
                    <option value="openrouter">OpenRouter</option>
                    <option value="vllm">vLLM</option>
                    <option value="litellm">LiteLLM</option>
                  </select>
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={() => removeOtherBackend(index)}
                    disabled={loading}
                  >
                    Remove
                  </Button>
                </div>
                <div className="space-y-2">
                  <label className="text-xs text-muted-foreground">Backend Kwargs:</label>
                  <KeyValueEditor
                    pairs={otherBackendKwargs[index] || []}
                    onChange={(pairs) => {
                      const newKwargs = [...otherBackendKwargs];
                      newKwargs[index] = pairs;
                      setOtherBackendKwargs(newKwargs);
                    }}
                    disabled={loading}
                  />
                </div>
              </div>
            ))}
          </div>

          {/* Verbose Output */}
          <div className="flex items-center space-x-2">
            <input
              id="verbose"
              type="checkbox"
              checked={verbose}
              onChange={(e) => setVerbose(e.target.checked)}
              className="w-4 h-4 rounded border-input"
              disabled={loading}
            />
            <label htmlFor="verbose" className="text-sm font-medium">
              Verbose Output (show detailed console output in response)
            </label>
          </div>

          {/* Enable Logging */}
          <div className="flex items-center space-x-2">
            <input
              id="enableLogging"
              type="checkbox"
              checked={enableLogging}
              onChange={(e) => setEnableLogging(e.target.checked)}
              className="w-4 h-4 rounded border-input"
              disabled={loading}
            />
            <label htmlFor="enableLogging" className="text-sm font-medium">
              Enable Logging (save to logs/)
            </label>
          </div>

          {/* Submit Button */}
          <Button
            type="submit"
            disabled={loading}
            className="w-full"
          >
            {loading ? 'Running...' : 'Run RLM'}
          </Button>
        </form>
      </CardContent>
    </Card>
  );
}

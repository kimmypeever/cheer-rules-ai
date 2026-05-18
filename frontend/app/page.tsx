"use client";

import { useState, useRef, useEffect } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Card, CardContent } from "@/components/ui/card";
import { Send, Loader2 } from "lucide-react";

type Source = { source: string; page: number; score: number };

type UserMessage = { role: "user"; content: string };
type AssistantMessage = {
  role: "assistant";
  content: string;
  sources: Source[];
  category: string | null;
};
type Message = UserMessage | AssistantMessage;

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const EXAMPLE_QUESTIONS = [
  "Is a basket toss legal at Level 3?",
  "What level can I do a helicopter?",
  "Can I load to prep without a spotter at Level 1?",
  "What tumbling is allowed at Level 2?",
];

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function submit(query: string) {
    if (!query.trim() || loading) return;
    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: query }]);
    setLoading(true);

    try {
      const res = await fetch(`${API_URL}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });
      const data = await res.json();
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: data.answer,
          sources: data.sources ?? [],
          category: data.category ?? null,
        },
      ]);
    } catch {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content:
            "Could not reach the rules engine. Make sure the API server is running:\n\n  uvicorn api:app --reload",
          sources: [],
          category: null,
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    submit(input.trim());
  }

  const dedupedSources = (sources: Source[]) =>
    [...new Map(sources.map((s) => [`${s.source}-${s.page}`, s])).values()];

  return (
    <div className="flex flex-col h-screen bg-slate-50">
      {/* Header */}
      <header className="bg-slate-900 text-white px-6 py-4 shrink-0">
        <h1 className="text-xl font-bold tracking-tight">Cheer Rules AI</h1>
        <p className="text-slate-400 text-sm">IASF / USASF rulebook assistant</p>
      </header>

      {/* Messages */}
      <ScrollArea className="flex-1 min-h-0 px-4 py-6">
        <div className="max-w-3xl mx-auto space-y-4">
          {messages.length === 0 && (
            <div className="text-center mt-20 space-y-6">
              <div className="space-y-1">
                <p className="text-slate-700 font-medium text-lg">Ask me anything about cheer rules</p>
                <p className="text-slate-400 text-sm">Skill legality, level requirements, sequences, definitions</p>
              </div>
              <div className="flex flex-wrap justify-center gap-2">
                {EXAMPLE_QUESTIONS.map((q) => (
                  <button
                    key={q}
                    onClick={() => submit(q)}
                    className="text-sm border border-slate-200 rounded-full px-4 py-1.5 text-slate-600 bg-white hover:bg-slate-50 hover:border-slate-300 transition-colors"
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((msg, i) =>
            msg.role === "user" ? (
              <div key={i} className="flex justify-end">
                <div className="bg-slate-900 text-white rounded-2xl rounded-tr-sm px-4 py-2.5 max-w-lg text-sm">
                  {msg.content}
                </div>
              </div>
            ) : (
              <div key={i} className="flex justify-start">
                <div className="max-w-2xl w-full space-y-2">
                  <Card className="border-slate-200 shadow-sm">
                    <CardContent className="pt-4 pb-3 px-4">
                      <pre className="text-sm whitespace-pre-wrap font-sans text-slate-800 leading-relaxed">
                        {msg.content}
                      </pre>
                    </CardContent>
                  </Card>
                  {msg.sources.length > 0 && (
                    <div className="flex flex-wrap items-center gap-1.5 px-1">
                      <span className="text-xs text-slate-400">Sources:</span>
                      {dedupedSources(msg.sources).map((s, j) => (
                        <Badge
                          key={j}
                          variant="secondary"
                          className="text-xs font-normal text-slate-500"
                        >
                          p.{s.page} · {Math.round(s.score * 100)}%
                        </Badge>
                      ))}
                      {msg.category && (
                        <Badge variant="outline" className="text-xs font-normal ml-auto text-slate-400">
                          {msg.category}
                        </Badge>
                      )}
                    </div>
                  )}
                </div>
              </div>
            )
          )}

          {loading && (
            <div className="flex justify-start">
              <Card className="border-slate-200 shadow-sm">
                <CardContent className="pt-4 pb-3 px-4">
                  <Loader2 className="h-4 w-4 animate-spin text-slate-400" />
                </CardContent>
              </Card>
            </div>
          )}

          <div ref={bottomRef} />
        </div>
      </ScrollArea>

      {/* Input bar */}
      <div className="shrink-0 border-t border-slate-200 bg-white px-4 py-3">
        <form onSubmit={handleSubmit} className="max-w-3xl mx-auto flex gap-2">
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about a skill, level, or sequence…"
            disabled={loading}
            className="flex-1"
          />
          <Button
            type="submit"
            disabled={loading || !input.trim()}
            className="bg-slate-900 hover:bg-slate-700 shrink-0"
          >
            <Send className="h-4 w-4" />
          </Button>
        </form>
      </div>
    </div>
  );
}

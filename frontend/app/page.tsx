"use client";

import { useState, useRef, useEffect } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Send, ThumbsUp, ThumbsDown, BookOpen, ArrowRight } from "lucide-react";

type Source = { source: string; page: number; score: number };

type UserMessage = { role: "user"; content: string };
type AssistantMessage = {
  role: "assistant";
  content: string;
  sources: Source[];
  category: string | null;
  rating?: "good" | "bad";
};
type Message = UserMessage | AssistantMessage;

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const EXAMPLE_QUESTIONS = [
  "Is a basket toss legal at Level 3?",
  "What level can I do a helicopter?",
  "Can I load to prep without a spotter at Level 1?",
  "What tumbling is allowed at Level 2?",
];

const CATEGORY_COLORS: Record<string, string> = {
  STUNTS: "bg-violet-100 text-violet-700",
  TOSSES: "bg-pink-100 text-pink-700",
  TUMBLING: "bg-orange-100 text-orange-700",
  PYRAMIDS: "bg-blue-100 text-blue-700",
  DISMOUNTS: "bg-teal-100 text-teal-700",
};

function answerAccent(content: string) {
  const first = content.trimStart().toUpperCase();
  if (first.startsWith("LEGAL")) return "border-l-green-400 bg-green-50/40";
  if (first.startsWith("ILLEGAL")) return "border-l-red-400 bg-red-50/40";
  return "border-l-violet-300 bg-white";
}

function BouncingDots() {
  return (
    <div className="flex gap-1.5 items-center px-1 py-1">
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="w-2 h-2 rounded-full bg-violet-400 animate-bounce"
          style={{ animationDelay: `${i * 0.15}s` }}
        />
      ))}
    </div>
  );
}

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
      if (!res.ok) {
        throw new Error(data.detail ?? "Unknown error");
      }
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: data.answer,
          sources: data.sources ?? [],
          category: data.category ?? null,
        },
      ]);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Unknown error";
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `The rules engine returned an error — please try again.\n\n${msg}`,
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

  async function rate(index: number, rating: "good" | "bad") {
    const msg = messages[index];
    if (msg.role !== "assistant" || msg.rating) return;
    setMessages((prev) =>
      prev.map((m, i) => (i === index && m.role === "assistant" ? { ...m, rating } : m))
    );
    const question = messages[index - 1]?.content ?? "";
    await fetch(`${API_URL}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, answer: msg.content, rating }),
    }).catch(() => {});
  }

  const dedupedSources = (sources: Source[]) =>
    [...new Map(sources.map((s) => [`${s.source}-${s.page}`, s])).values()];

  return (
    <div className="flex flex-col h-screen bg-violet-50">

      {/* Header */}
      <header className="shrink-0 bg-gradient-to-r from-violet-600 to-indigo-600 text-white px-6 py-4 shadow-lg">
        <div className="max-w-3xl mx-auto flex items-center gap-3">
          <div className="bg-white/15 rounded-xl p-2">
            <BookOpen className="h-5 w-5 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight">Cheer Rules AI</h1>
            <p className="text-violet-200 text-xs font-medium tracking-wide uppercase">IASF / USASF Rulebook Assistant</p>
          </div>
        </div>
      </header>

      {/* Messages */}
      <ScrollArea className="flex-1 min-h-0 px-4 py-6">
        <div className="max-w-3xl mx-auto space-y-5">

          {messages.length === 0 && (
            <div className="text-center mt-16 space-y-8">
              <div className="space-y-2">
                <p className="text-2xl font-bold text-violet-900">What do you want to know?</p>
                <p className="text-violet-400 text-sm">Skill legality · Level requirements · Sequences · Definitions</p>
              </div>
              <div className="flex flex-wrap justify-center gap-2.5">
                {EXAMPLE_QUESTIONS.map((text) => (
                  <button
                    key={text}
                    onClick={() => submit(text)}
                    className="group flex items-center gap-2 text-sm border border-violet-200 rounded-xl px-4 py-2.5 text-violet-700 bg-white font-medium hover:border-violet-400 hover:bg-violet-50 hover:shadow-sm transition-all"
                  >
                    {text}
                    <ArrowRight className="h-3.5 w-3.5 text-violet-400 opacity-0 -translate-x-1 group-hover:opacity-100 group-hover:translate-x-0 transition-all" />
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((msg, i) =>
            msg.role === "user" ? (
              <div key={i} className="flex justify-end">
                <div className="bg-gradient-to-br from-violet-600 to-indigo-600 text-white rounded-2xl rounded-tr-sm px-4 py-2.5 max-w-lg text-sm shadow-md font-medium">
                  {msg.content}
                </div>
              </div>
            ) : (
              <div key={i} className="flex justify-start">
                <div className="max-w-2xl w-full space-y-2">
                  <div
                    className={`rounded-2xl rounded-tl-sm border-l-4 shadow-sm px-4 pt-4 pb-3 ${answerAccent(msg.content)}`}
                  >
                    <pre className="text-sm whitespace-pre-wrap font-sans text-slate-800 leading-relaxed">
                      {msg.content}
                    </pre>
                  </div>

                  {/* Footer row: sources + category + feedback */}
                  <div className="flex flex-wrap items-center gap-2 px-1">
                    {dedupedSources(msg.sources).map((s, j) => (
                      <span
                        key={j}
                        className="text-xs bg-white border border-violet-100 text-violet-500 rounded-full px-2.5 py-0.5 font-medium shadow-sm"
                      >
                        p.{s.page} · {Math.round(s.score * 100)}%
                      </span>
                    ))}
                    {msg.category && (
                      <span
                        className={`text-xs rounded-full px-2.5 py-0.5 font-semibold ${
                          CATEGORY_COLORS[msg.category] ?? "bg-slate-100 text-slate-500"
                        }`}
                      >
                        {msg.category}
                      </span>
                    )}

                    <div className="ml-auto flex items-center gap-1">
                      <button
                        onClick={() => rate(i, "good")}
                        disabled={!!msg.rating}
                        title="Good answer"
                        className={`p-1.5 rounded-full transition-all ${
                          msg.rating === "good"
                            ? "bg-green-100 text-green-500"
                            : msg.rating
                            ? "text-slate-200 cursor-default"
                            : "text-slate-300 hover:bg-green-50 hover:text-green-500"
                        }`}
                      >
                        <ThumbsUp className="h-3.5 w-3.5" />
                      </button>
                      <button
                        onClick={() => rate(i, "bad")}
                        disabled={!!msg.rating}
                        title="Bad answer"
                        className={`p-1.5 rounded-full transition-all ${
                          msg.rating === "bad"
                            ? "bg-red-100 text-red-400"
                            : msg.rating
                            ? "text-slate-200 cursor-default"
                            : "text-slate-300 hover:bg-red-50 hover:text-red-400"
                        }`}
                      >
                        <ThumbsDown className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            )
          )}

          {loading && (
            <div className="flex justify-start">
              <div className="bg-white border-l-4 border-l-violet-300 rounded-2xl rounded-tl-sm shadow-sm px-4 py-3">
                <BouncingDots />
              </div>
            </div>
          )}

          <div ref={bottomRef} />
        </div>
      </ScrollArea>

      {/* Input bar */}
      <div className="shrink-0 border-t-2 border-violet-100 bg-white px-4 py-3">
        <form onSubmit={handleSubmit} className="max-w-3xl mx-auto flex gap-2">
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about a skill, level, or sequence…"
            disabled={loading}
            className="flex-1 border-2 border-violet-200 focus-visible:ring-violet-400 rounded-xl"
          />
          <Button
            type="submit"
            disabled={loading || !input.trim()}
            className="bg-gradient-to-r from-violet-600 to-indigo-600 hover:from-violet-700 hover:to-indigo-700 text-white rounded-xl shrink-0 shadow-md"
          >
            <Send className="h-4 w-4" />
          </Button>
        </form>
      </div>
    </div>
  );
}

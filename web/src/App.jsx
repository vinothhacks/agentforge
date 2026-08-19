import React, { useEffect, useState } from "react";

export default function App() {
  const [ws, setWs] = useState("");
  const [key, setKey] = useState("");
  const [log, setLog] = useState([]);
  const [msg, setMsg] = useState("");
  const [meter, setMeter] = useState("tokens 0");
  const [session, setSession] = useState("default");

  useEffect(() => {
    fetch("/api/health")
      .then((r) => r.json())
      .then((h) => setWs(h.workspace));
  }, []);

  async function send(e) {
    e.preventDefault();
    const text = msg.trim();
    if (!text) return;
    setMsg("");
    setLog((l) => [...l, { role: "user", text }]);
    const r = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, session_id: session }),
    });
    const data = await r.json();
    if (!r.ok) {
      setLog((l) => [...l, { role: "bot", text: JSON.stringify(data) }]);
      return;
    }
    setSession(data.session_id);
    setLog((l) => [...l, { role: "bot", text: data.text }]);
    const u = data.usage || {};
    setMeter(`tokens ${u.prompt_tokens || 0} / ${u.max_tokens} · $${(u.usd || 0).toFixed(4)}`);
  }

  return (
    <div style={{ fontFamily: "Segoe UI, sans-serif", background: "#0f1419", color: "#e8eef4", minHeight: "100vh" }}>
      <header style={{ padding: 12, borderBottom: "1px solid #2a3542" }}>
        <strong>AgentForge · PDA query</strong>
        <div style={{ opacity: 0.6, fontSize: 12 }}>{ws}</div>
        <div style={{ fontSize: 12 }}>{meter}</div>
      </header>
      <aside style={{ padding: 12 }}>
        <input type="password" value={key} onChange={(e) => setKey(e.target.value)} placeholder="OpenRouter key" />
        <button
          onClick={() =>
            fetch("/api/key", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ key }),
            })
          }
        >
          Save key
        </button>
        <p style={{ color: "#e0a14a", fontSize: 12 }}>Export to Excel comes next.</p>
      </aside>
      <main style={{ padding: 16 }}>
        {log.map((m, i) => (
          <pre key={i} style={{ background: "#1a222c", padding: 10 }}>
            {m.role}: {m.text}
          </pre>
        ))}
        <form onSubmit={send}>
          <textarea value={msg} onChange={(e) => setMsg(e.target.value)} placeholder="Which PDAs mention demurrage?" />
          <button type="submit">Ask</button>
        </form>
      </main>
    </div>
  );
}

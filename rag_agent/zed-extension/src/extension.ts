/**
 * Minimal Zed extension skeleton (TypeScript)
 *
 * Purpose:
 * - Provide a small, safe fallback skeleton to call a local RAG agent service.
 * - Works as a starting point for a real Zed extension; the Zed extension API changes over time,
 *   so this file intentionally uses flexible, defensive calls against `context`.
 *
 * Usage:
 * - The extension manifest should reference the built output (e.g. `dist/extension.js`).
 * - Configure the agent base URL via environment variable `RAG_AGENT_BASE` or change the default below.
 *
 * Notes:
 * - This file is intentionally implementation-agnostic: it attempts to use common extension host
 *   primitives if present on `context`, and falls back to CLI behavior when executed with Node.
 * - The allow/deny UI and patch application are left for future enhancement; this skeleton returns
 *   the agent's response and logs it, and provides hooks where you can implement Zed-specific UI.
 */

const AGENT_BASE = (typeof process !== "undefined" && process.env && process.env.RAG_AGENT_BASE) || "http://127.0.0.1:7860";

/**
 * callAgentChat
 * Simple helper that posts a chat request to the local agent service.
 * Expects the agent to respond with an envelope { retrieved: [...], proxy_response: {...} }.
 */
async function callAgentChat(prompt: string, top_k = 3) {
  try {
    const body = {
      messages: [{ role: "user", content: prompt }],
      top_k,
    };

    const res = await fetch(`${AGENT_BASE.replace(/\/+$/, "")}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      const text = await res.text();
      throw new Error(`Agent returned ${res.status}: ${text}`);
    }

    const json = await res.json();
    return json;
  } catch (err) {
    // bubble error to caller
    throw err;
  }
}

/**
 * promptForInput
 * Attempts to use extension host UI (if available) to ask the user for a prompt.
 * Falls back to a simple Node stdin prompt for CLI testing.
 */
async function promptForInput(context: any, promptText: string): Promise<string | undefined> {
  // Preferred: use context.window.showInputBox or similar if available
  try {
    if (context && context.window && typeof context.window.showInputBox === "function") {
      // Zed or host environment may provide showInputBox
      return await context.window.showInputBox({ prompt: promptText });
    }

    // Another common shape: context.commands.registerCommand environment may expose UI helpers
    if (context && typeof context.prompt === "function") {
      return await context.prompt(promptText);
    }
  } catch (e) {
    // ignore and fall through to CLI fallback
  }

  // CLI fallback: read from stdin (synchronous-ish)
  if (typeof process !== "undefined" && process.stdin) {
    return await new Promise((resolve) => {
      process.stdout.write(`${promptText} `);
      process.stdin.resume();
      process.stdin.setEncoding("utf8");
      process.stdin.once("data", function (data) {
        process.stdin.pause();
        resolve(data.toString().trim());
      });
    });
  }

  // If nothing available, return undefined
  return undefined;
}

/**
 * showResult
 * Attempts to show a message to the user using the extension host UI; falls back to console.log.
 */
async function showResult(context: any, title: string, body: string) {
  try {
    if (context && context.window && typeof context.window.showInformationMessage === "function") {
      await context.window.showInformationMessage(`${title}\n\n${body}`);
      return;
    }
    if (context && typeof context.notify === "function") {
      await context.notify(`${title}: ${body}`);
      return;
    }
  } catch (e) {
    // ignore and fall back
  }

  // CLI / generic fallback
  console.log("=== " + title + " ===");
  console.log(body);
  console.log("=== end ===");
}

/**
 * registerCommandHelper
 * Registers a command with the extension host if the API is available.
 * The signature and placement varies between editor extension APIs; we attempt common locations.
 */
function registerCommandHelper(context: any, commandId: string, handler: (...args: any[]) => any) {
  // Try common registration patterns
  try {
    // pattern: context.subscriptions.registerCommand(...)
    if (context && context.subscriptions && typeof context.subscriptions.registerCommand === "function") {
      return context.subscriptions.registerCommand(commandId, handler);
    }

    // pattern: context.commands.registerCommand(...)
    if (context && context.commands && typeof context.commands.registerCommand === "function") {
      return context.commands.registerCommand(commandId, handler);
    }

    // pattern: global registration object
    if (context && typeof context.registerCommand === "function") {
      return context.registerCommand(commandId, handler);
    }
  } catch (e) {
    // swallow - host may differ
  }

  // If registration is not available, return a noop disposable
  return {
    dispose: () => {},
  };
}

/**
 * activate
 * Entry point called by the host (Zed) when the extension is activated.
 * We attempt to register a command `zed-rag-agent.ask` that prompts the user, calls the agent,
 * and shows the result. Replace UI integration points with Zed-specific APIs as needed.
 */
export async function activate(context: any) {
  // Defensive log
  try {
    if (context && context.logger && typeof context.logger.info === "function") {
      context.logger.info("Activating zed-rag-agent extension (skeleton)");
    } else {
      console.info("Activating zed-rag-agent extension (skeleton)");
    }
  } catch (e) {
    // ignore
  }

  // Register the command with the host
  const disposable = registerCommandHelper(context, "zed-rag-agent.ask", async () => {
    try {
      const prompt = await promptForInput(context, "Question for RAG agent:");
      if (!prompt) {
        await showResult(context, "Cancelled", "No prompt provided.");
        return;
      }

      // Call agent
      const envelope = await callAgentChat(prompt, /* top_k */ 3);

      // Extract friendly assistant text if present (OpenAI-style)
      let assistantText = "";
      try {
        if (envelope && envelope.proxy_response) {
          // many proxies return { choices: [{ message: { content } }] }
          const choices = envelope.proxy_response.choices;
          if (Array.isArray(choices) && choices.length > 0) {
            assistantText = (choices[0].message && choices[0].message.content) || choices[0].text || JSON.stringify(choices[0]);
          } else {
            assistantText = JSON.stringify(envelope.proxy_response, null, 2);
          }
        } else {
          assistantText = JSON.stringify(envelope, null, 2);
        }
      } catch (e) {
        assistantText = JSON.stringify(envelope, null, 2);
      }

      // Show to user (modal / info)
      await showResult(context, "RAG Agent Response", assistantText);

      // NOTE: In a full implementation, you would parse `envelope.retrieved` and any
      // structured plan or patch returned by the agent, then show a rich allow/deny UI,
      // and on Allow apply the patch via Zed's workspace/file APIs.
    } catch (err: any) {
      const msg = err && err.message ? err.message : String(err);
      await showResult(context, "RAG Agent Error", msg);
    }
  });

  // store disposable if host provides subscriptions
  try {
    if (context && context.subscriptions && typeof context.subscriptions.push === "function") {
      context.subscriptions.push(disposable);
    }
  } catch (e) {
    // ignore
  }
}

/**
 * deactivate
 * Host calls this when the extension is unloaded. Clean up resources if needed.
 */
export function deactivate() {
  // Nothing to clean up in the skeleton; real implementations should dispose handles here.
  try {
    if (typeof console !== "undefined" && console.info) {
      console.info("Deactivating zed-rag-agent extension (skeleton)");
    }
  } catch (e) {
    // ignore
  }
}

/**
 * If this file is executed directly with Node (for CLI debugging), provide a tiny runner.
 * Example:
 *   node dist/extension.js "Summarize the following function..."
 */
if (typeof require !== "undefined" && require.main === module) {
  (async () => {
    try {
      const prompt = process.argv.slice(2).join(" ") || "Summarize the selected function and list edge cases.";
      const res = await callAgentChat(prompt, 3);
      console.log("Agent envelope:\n", JSON.stringify(res, null, 2));
      process.exit(0);
    } catch (err: any) {
      console.error("Agent call failed:", err && err.message ? err.message : err);
      process.exit(2);
    }
  })();
}

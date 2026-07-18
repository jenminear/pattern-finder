// Pattern Finder frontend. Talks to the agent over the real A2A protocol
// (not a bespoke REST endpoint) via the official @a2a-js/sdk client, since
// A2UI's DataParts ride the A2A wire format -- see .agents-cli-spec.md,
// "UI Surfaces (A2UI)".

import { A2AClient } from "@a2a-js/sdk/client";
import { v4 as uuidv4 } from "uuid";

const NUM_INPUT_ROWS = 5;
// ClientFactory.createFromUrl()'s path-resolution convention didn't match
// our nested RPC path (/a2a/pattern-finder, not domain root) -- using
// A2AClient.fromCardUrl() with the exact, known-correct card URL instead.
const AGENT_CARD_URL = `${window.location.origin}/a2a/pattern-finder/.well-known/agent-card.json`;

// Mirrors app/agent.py's _ECONOMY/_BALANCED/_MAX_QUALITY tiers -- all three
// now share one model (gemini-3.5-flash and gemini-2.5-pro were dropped
// after a billing review; see app/agent.py's effort-tier comment), so this
// is a flat constant rather than a dial-position lookup. Kept as a
// function/named constant (not inlined) so a future re-introduction of
// per-tier models only needs to change this one spot -- must still be kept
// in sync by hand with app/agent.py's _FLASH_LITE.
const EFFORT_TIER_MODEL = "gemini-3.1-flash-lite";

function effortTierModel(_dial) {
  return EFFORT_TIER_MODEL;
}

const inputRowsEl = document.getElementById("input-rows");
const effortDialEl = document.getElementById("effort-dial");
const effortDialModelEl = document.getElementById("effort-dial-model");
const emitUiToggleEl = document.getElementById("emit-ui-toggle");
const guessBtn = document.getElementById("guess-btn");
const estimateBoxEl = document.getElementById("estimate-box");
const reasoningCardEl = document.getElementById("reasoning-card");
const correctAnswerFieldEl = document.getElementById("correct-answer-field");
const correctConsequenceEl = document.getElementById("correct-consequence");
const submitAnswerBtn = document.getElementById("submit-answer-btn");
const capturedCardEl = document.getElementById("captured-card");
const newScenarioBtn = document.getElementById("new-scenario-btn");

const scorecard = { processed: 0, dontKnow: 0, correct: 0, incorrect: 0 };

// --- Build the 5 label/value input rows -------------------------------
for (let i = 0; i < NUM_INPUT_ROWS; i++) {
  const row = document.createElement("div");
  row.className = "input-row";
  row.innerHTML = `
    <input type="text" class="label-input" placeholder="label ${i + 1}" maxlength="100" />
    <input type="text" class="value-input" placeholder="value ${i + 1}" maxlength="300" />
  `;
  inputRowsEl.appendChild(row);
}

function updateEffortDialModel() {
  effortDialModelEl.textContent = effortTierModel(parseFloat(effortDialEl.value));
}
updateEffortDialModel();
effortDialEl.addEventListener("input", updateEffortDialModel);

// --- A2A client ---------------------------------------------------------
let clientPromise = null;
function getClient() {
  if (!clientPromise) {
    clientPromise = A2AClient.fromCardUrl(AGENT_CARD_URL);
  }
  return clientPromise;
}

let sessionContextId = null;
// Everything the Learn-phase message needs to echo back, since the
// deterministic backend orchestrator has no access to "what the agent was
// thinking" during the prior guess turn -- see app/agent.py's module
// docstring. Populated from the Agent Reasoning card's payload.
let pendingScenario = null; // [[label, value], ...] from the guess turn
let pendingGuess = null; // { guess, matchedVia, patternId, rule } or null

function collectInputs() {
  const rows = [...inputRowsEl.querySelectorAll(".input-row")];
  const pairs = [];
  for (const row of rows) {
    const label = row.querySelector(".label-input").value.trim();
    const value = row.querySelector(".value-input").value.trim();
    if (label) pairs.push([label, value]);
  }
  return pairs;
}

function scenarioLines(pairs) {
  return pairs.map(([label, value]) => `label=${label} value=${value}`);
}

function protocolHeader() {
  const dial = effortDialEl.value;
  const emitUi = emitUiToggleEl.checked ? "on" : "off";
  return `EFFORT_DIAL: ${dial}\nEMIT_UI: ${emitUi}`;
}

// Extracts (text, dataParts[]) from whatever shape the streamed/awaited A2A
// response turns out to have -- kept defensive since this is the newest,
// least-documented part of the stack.
function extractParts(parts) {
  let text = "";
  const dataParts = [];
  for (const part of parts || []) {
    if (part.kind === "text" && part.text) text += part.text;
    else if (part.kind === "data" && part.data) dataParts.push(part.data);
  }
  return { text, dataParts };
}

async function sendA2AMessage(text) {
  const client = await getClient();
  const message = {
    messageId: uuidv4(),
    contextId: sessionContextId ?? undefined,
    role: "user",
    parts: [{ kind: "text", text }],
    kind: "message",
  };

  let combinedText = "";
  const allDataParts = [];

  const stream = client.sendMessageStream({ message });
  for await (const event of stream) {
    console.debug("[a2a event]", event);
    if (event.contextId && !sessionContextId) sessionContextId = event.contextId;
    if (event.kind === "message") {
      const { text: t, dataParts } = extractParts(event.parts);
      combinedText += t;
      allDataParts.push(...dataParts);
    } else if (event.kind === "artifact-update" && event.artifact) {
      const { text: t, dataParts } = extractParts(event.artifact.parts);
      combinedText += t;
      allDataParts.push(...dataParts);
    } else if (event.kind === "task" && event.artifacts) {
      for (const artifact of event.artifacts) {
        const { text: t, dataParts } = extractParts(artifact.parts);
        combinedText += t;
        allDataParts.push(...dataParts);
      }
    } else if (event.status?.message?.parts) {
      const { text: t, dataParts } = extractParts(event.status.message.parts);
      combinedText += t;
      allDataParts.push(...dataParts);
    }
  }

  return { text: combinedText, dataParts: allDataParts };
}

function findSurface(dataParts, surface) {
  return dataParts.find((d) => d.surface === surface) || null;
}

function setBusy(isBusy, button) {
  button.disabled = isBusy;
  button.textContent = isBusy ? "Thinking..." : button.dataset.label;
}

// --- Guess -----------------------------------------------------------
guessBtn.dataset.label = guessBtn.textContent;
guessBtn.addEventListener("click", async () => {
  const pairs = collectInputs();
  if (pairs.length === 0) {
    alert("Enter at least one label/value pair.");
    return;
  }

  sessionContextId = null; // new scenario -> new A2A context
  pendingScenario = pairs;
  pendingGuess = null;
  const messageText = `${protocolHeader()}\nPHASE: guess\n${scenarioLines(pairs).join("\n")}`;

  setBusy(true, guessBtn);
  estimateBoxEl.textContent = "...";
  reasoningCardEl.hidden = true;
  capturedCardEl.hidden = true;
  correctAnswerFieldEl.hidden = true;

  try {
    const { text, dataParts } = await sendA2AMessage(messageText);
    const reasoning = findSurface(dataParts, "agent_reasoning");

    if (reasoning) {
      pendingGuess = {
        guess: reasoning.guess ?? null,
        matchedVia: reasoning.matched_via ?? null,
        patternId: reasoning.pattern_id ?? null,
        rule: reasoning.rule ?? null,
      };
      estimateBoxEl.textContent = reasoning.guess ?? "I don't know";
      reasoningCardEl.payload = reasoning;
      reasoningCardEl.hidden = false;
    } else {
      // Fall back to the raw text reply if the DataPart didn't arrive
      // (e.g. "Show agent reasoning details" is off) -- still usable, the
      // Learn-phase turn just won't be able to attribute a pattern precisely.
      pendingGuess = { guess: text.trim(), matchedVia: null, patternId: null, rule: null };
      estimateBoxEl.textContent = text || "(no response)";
    }
    correctAnswerFieldEl.hidden = false;
  } catch (err) {
    console.error(err);
    estimateBoxEl.textContent = `Error: ${err.message || err}`;
  } finally {
    setBusy(false, guessBtn);
  }
});

// --- Learn -------------------------------------------------------------
submitAnswerBtn.dataset.label = submitAnswerBtn.textContent;
submitAnswerBtn.addEventListener("click", async () => {
  const correct = correctConsequenceEl.value.trim();
  if (!correct) {
    alert("Enter the correct consequence.");
    return;
  }
  if (!sessionContextId || !pendingScenario) {
    alert("Guess a scenario first.");
    return;
  }

  const lines = [
    protocolHeader(),
    "PHASE: learn",
    `CORRECT_CONSEQUENCE: ${correct}`,
    ...scenarioLines(pendingScenario), // deterministic backend re-derives insert_scenario from these
  ];
  if (pendingGuess?.guess != null) lines.push(`GUESS_VALUE: ${pendingGuess.guess}`);
  if (pendingGuess?.matchedVia) lines.push(`MATCHED_VIA: ${pendingGuess.matchedVia}`);
  if (pendingGuess?.patternId != null) lines.push(`PATTERN_ID: ${pendingGuess.patternId}`);
  if (pendingGuess?.rule) lines.push(`APPLIED_RULE: ${pendingGuess.rule}`);
  const messageText = lines.join("\n");

  setBusy(true, submitAnswerBtn);

  try {
    const { dataParts } = await sendA2AMessage(messageText);
    const captured = findSurface(dataParts, "pattern_captured");

    scorecard.processed += 1;
    const guessNorm = (pendingGuess?.guess || "").trim().toLowerCase();
    const isDontKnow = !guessNorm || guessNorm.includes("don't know") || guessNorm.includes("dont know");
    if (isDontKnow) {
      scorecard.dontKnow += 1;
    } else if (captured?.matched) {
      scorecard.correct += 1;
    } else {
      scorecard.incorrect += 1;
    }
    renderScorecard();

    if (captured && captured.action && captured.action !== "none") {
      capturedCardEl.payload = captured;
      capturedCardEl.hidden = false;
    }
    correctAnswerFieldEl.hidden = true;
  } catch (err) {
    console.error(err);
    alert(`Error recording the answer: ${err.message || err}`);
  } finally {
    setBusy(false, submitAnswerBtn);
  }
});

// --- Reset ---------------------------------------------------------------
newScenarioBtn.addEventListener("click", () => {
  sessionContextId = null;
  pendingScenario = null;
  pendingGuess = null;
  inputRowsEl.querySelectorAll("input").forEach((el) => (el.value = ""));
  correctConsequenceEl.value = "";
  estimateBoxEl.textContent = "—";
  reasoningCardEl.hidden = true;
  capturedCardEl.hidden = true;
  correctAnswerFieldEl.hidden = true;
});

function renderScorecard() {
  const pct = (n) => (scorecard.processed ? `${Math.round((n / scorecard.processed) * 100)}%` : "—");
  document.getElementById("sc-processed").textContent = scorecard.processed;
  document.getElementById("sc-dontknow").textContent = scorecard.dontKnow;
  document.getElementById("sc-dontknow-pct").textContent = pct(scorecard.dontKnow);
  document.getElementById("sc-correct").textContent = scorecard.correct;
  document.getElementById("sc-correct-pct").textContent = pct(scorecard.correct);
  document.getElementById("sc-incorrect").textContent = scorecard.incorrect;
  document.getElementById("sc-incorrect-pct").textContent = pct(scorecard.incorrect);
}

// A2UI-rendered surfaces (.agents-cli-spec.md, "UI Surfaces (A2UI)").
// Lit is the only fully-supported official A2UI renderer today (React
// support is roadmap-only) -- see the design discussion in project memory.
// These two custom elements render the JSON payloads the agent emits via
// its emit_agent_reasoning / emit_pattern_captured tools, delivered as A2A
// DataParts (catalog IDs "agent_reasoning" / "pattern_captured").

import { LitElement, html, css } from "lit";

const MATCHED_VIA_LABELS = {
  exact_pattern: "an existing pattern (exact label match)",
  pattern_search: "a new pattern found via search",
  label_independent_match: "a pattern matched independent of labels",
  none: "no pattern",
};

export class AgentReasoningCard extends LitElement {
  static properties = {
    payload: { type: Object },
  };

  static styles = css`
    :host([hidden]) {
      display: none;
    }
    :host {
      display: block;
      border: 1px solid var(--border, #ddd);
      border-radius: 8px;
      padding: 1rem 1.25rem;
      margin: 0.75rem 0;
      background: var(--surface, #f7f7fb);
    }
    h3 {
      margin: 0 0 0.5rem;
      font-size: 0.95rem;
      text-transform: uppercase;
      letter-spacing: 0.03em;
      color: var(--accent, #4b3fb0);
    }
    dl {
      display: grid;
      grid-template-columns: max-content 1fr;
      gap: 0.25rem 0.75rem;
      margin: 0 0 0.5rem;
    }
    dt { font-weight: 600; }
    dd { margin: 0; }
    .trace {
      font-size: 0.9rem;
      color: #444;
      border-top: 1px solid var(--border, #ddd);
      padding-top: 0.5rem;
      margin-top: 0.5rem;
      white-space: pre-wrap;
    }
  `;

  render() {
    if (!this.payload) return html``;
    const { guess, confidence, matched_via, trace } = this.payload;
    const confidencePct = confidence != null ? `${Math.round(confidence * 100)}%` : "?";
    const matchedLabel = MATCHED_VIA_LABELS[matched_via] || matched_via || "unknown";
    return html`
      <h3>Agent Reasoning</h3>
      <dl>
        <dt>Guess</dt><dd>${guess ?? "I don't know"}</dd>
        <dt>Confidence</dt><dd>${confidencePct}</dd>
        <dt>Matched via</dt><dd>${matchedLabel}</dd>
      </dl>
      ${trace ? html`<div class="trace">${trace}</div>` : ""}
    `;
  }
}

const ACTION_LABELS = {
  created: "New pattern created",
  updated_label_set: "Existing pattern's label set updated",
  linked_only: "Linked to this pattern (label set unchanged)",
};

export class PatternCapturedCard extends LitElement {
  static properties = {
    payload: { type: Object },
  };

  static styles = css`
    :host([hidden]) {
      display: none;
    }
    :host {
      display: block;
      border: 1px solid var(--border, #ddd);
      border-radius: 8px;
      padding: 1rem 1.25rem;
      margin: 0.75rem 0;
      background: var(--surface-accent, #f0f7f0);
    }
    h3 {
      margin: 0 0 0.5rem;
      font-size: 0.95rem;
      text-transform: uppercase;
      letter-spacing: 0.03em;
      color: var(--accent-2, #2f7d3c);
    }
    dl {
      display: grid;
      grid-template-columns: max-content 1fr;
      gap: 0.25rem 0.75rem;
      margin: 0;
    }
    dt { font-weight: 600; }
    dd { margin: 0; }
    code {
      background: rgba(0, 0, 0, 0.06);
      padding: 0.1rem 0.35rem;
      border-radius: 4px;
    }
  `;

  render() {
    if (!this.payload) return html``;
    const { action, text_desc, rule_or_code_link, scenarios_linked } = this.payload;
    return html`
      <h3>Pattern Captured</h3>
      <dl>
        <dt>Action</dt><dd>${ACTION_LABELS[action] || action}</dd>
        ${text_desc ? html`<dt>Description</dt><dd>${text_desc}</dd>` : ""}
        ${rule_or_code_link
          ? html`<dt>Rule</dt><dd><code>${rule_or_code_link}</code></dd>`
          : ""}
        <dt>Scenarios linked</dt><dd>${scenarios_linked ?? 0}</dd>
      </dl>
    `;
  }
}

customElements.define("agent-reasoning-card", AgentReasoningCard);
customElements.define("pattern-captured-card", PatternCapturedCard);

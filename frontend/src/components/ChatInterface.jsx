import { useState, useEffect, useRef } from 'react';
import Stage1 from './Stage1';
import Stage2 from './Stage2';
import Stage3 from './Stage3';
import MarkdownRenderer from './MarkdownRenderer';
import './ChatInterface.css';

const STAGE_LABELS = {
  stage1: 'Stage 1',
  stage2: 'Stage 2',
  stage3: 'Stage 3',
};

function flattenStageItems(stageMap = {}) {
  return Object.entries(stageMap).flatMap(([stageKey, items]) =>
    (items || []).map((item) => ({
      stageKey,
      ...item,
    }))
  );
}

function formatFailureReason(failure) {
  if (failure?.status_code === 429) {
    return 'Rate limited on free tier (HTTP 429)';
  }

  if (failure?.status_code) {
    return `HTTP ${failure.status_code}`;
  }

  return 'Request failed';
}

function ModelAvailability({ metadata }) {
  const rateLimitItems = flattenStageItems(metadata?.rate_limits);
  const failureItems = flattenStageItems(metadata?.failures);
  const fallbackItems = flattenStageItems(metadata?.fallbacks);

  if (rateLimitItems.length === 0 && failureItems.length === 0 && fallbackItems.length === 0) {
    return null;
  }

  return (
    <div className="model-availability-panel">
      <h4>Model Availability</h4>

      {rateLimitItems.length > 0 && (
        <div className="model-availability-section model-availability-rate-limit">
          <strong>Rate limit exceeded (HTTP 429)</strong>
          <ul>
            {rateLimitItems.map((item, index) => {
              const requestedModel = item.requested_model || 'unknown model';
              const usedModel = item.used_model;
              const eventCount = item.event_count || 1;
              return (
                <li key={`rate-limit-${index}`}>
                  <span className="availability-stage">
                    {STAGE_LABELS[item.stageKey] || item.stageKey}:
                  </span>{' '}
                  <span className="availability-model">{requestedModel}</span>{' '}
                  <span className="availability-reason">
                    (rate limit exceeded {eventCount}x{usedModel && usedModel !== requestedModel ? `, rerouted to ${usedModel}` : ''})
                  </span>
                </li>
              );
            })}
          </ul>
        </div>
      )}

      {failureItems.length > 0 && (
        <div className="model-availability-section">
          <strong>Failed models</strong>
          <ul>
            {failureItems.map((failure, index) => (
              <li key={`failure-${index}`}>
                <span className="availability-stage">
                  {STAGE_LABELS[failure.stageKey] || failure.stageKey}:
                </span>{' '}
                <span className="availability-model">{failure.model}</span>{' '}
                <span className="availability-reason">({formatFailureReason(failure)})</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {fallbackItems.length > 0 && (
        <div className="model-availability-section">
          <strong>Fallback reroutes</strong>
          <ul>
            {fallbackItems.map((fallback, index) => (
              <li key={`fallback-${index}`}>
                <span className="availability-stage">
                  {STAGE_LABELS[fallback.stageKey] || fallback.stageKey}:
                </span>{' '}
                <span className="availability-model">{fallback.requested_model}</span>{' '}
                <span className="availability-reason">-&gt; {fallback.used_model}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default function ChatInterface({
  conversation,
  onSendMessage,
  isLoading,
}) {
  const [input, setInput] = useState('');
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [conversation]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (input.trim() && !isLoading) {
      onSendMessage(input);
      setInput('');
    }
  };

  const handleKeyDown = (e) => {
    // Submit on Enter (without Shift)
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  if (!conversation) {
    return (
      <div className="chat-interface">
        <div className="empty-state">
          <h2>Welcome to LLM Council</h2>
          <p>Create a new conversation to get started</p>
        </div>
      </div>
    );
  }

  return (
    <div className="chat-interface">
      <div className="messages-container">
        {conversation.messages.length === 0 ? (
          <div className="empty-state">
            <h2>Start a conversation</h2>
            <p>Ask a question to consult the LLM Council</p>
          </div>
        ) : (
          conversation.messages.map((msg, index) => (
            <div key={index} className="message-group">
              {msg.role === 'user' ? (
                <div className="user-message">
                  <div className="message-label">You</div>
                  <div className="message-content">
                    <div className="markdown-content">
                      <MarkdownRenderer content={msg.content} />
                    </div>
                  </div>
                </div>
              ) : (
                <div className="assistant-message">
                  <div className="message-label">LLM Council</div>
                  <ModelAvailability metadata={msg.metadata} />

                  {/* Stage 1 */}
                  {msg.loading?.stage1 && (
                    <div className="stage-loading">
                      <div className="spinner"></div>
                      <span>Running Stage 1: Collecting individual responses...</span>
                    </div>
                  )}
                  {msg.stage1 && (
                    <Stage1
                      responses={msg.stage1}
                      expectedCount={msg.metadata?.requested_models?.length}
                    />
                  )}

                  {/* Stage 2 */}
                  {msg.loading?.stage2 && (
                    <div className="stage-loading">
                      <div className="spinner"></div>
                      <span>Running Stage 2: Peer rankings...</span>
                    </div>
                  )}
                  {msg.stage2 && (
                    <Stage2
                      rankings={msg.stage2}
                      labelToModel={msg.metadata?.label_to_model}
                      aggregateRankings={msg.metadata?.aggregate_rankings}
                    />
                  )}

                  {/* Stage 3 */}
                  {msg.loading?.stage3 && (
                    <div className="stage-loading">
                      <div className="spinner"></div>
                      <span>Running Stage 3: Final synthesis...</span>
                    </div>
                  )}
                  {msg.stage3 && <Stage3 finalResponse={msg.stage3} />}
                </div>
              )}
            </div>
          ))
        )}

        {isLoading && (
          <div className="loading-indicator">
            <div className="spinner"></div>
            <span>Consulting the council...</span>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {conversation.messages.length === 0 && (
        <form className="input-form" onSubmit={handleSubmit}>
          <textarea
            className="message-input"
            placeholder="Ask your question... (Shift+Enter for new line, Enter to send)"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isLoading}
            rows={3}
          />
          <button
            type="submit"
            className="send-button"
            disabled={!input.trim() || isLoading}
          >
            Send
          </button>
        </form>
      )}
    </div>
  );
}

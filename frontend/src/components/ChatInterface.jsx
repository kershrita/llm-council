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

const VIEW_MODE_STAGES = 'stages';

const FRIENDLY_STATUS_STEPS = [
  {
    key: 'understanding',
    label: 'Understanding your question',
    description: 'Interpreting your request and context.',
  },
  {
    key: 'approach',
    label: 'Selecting approach',
    description: 'Planning how the council will tackle it.',
  },
  {
    key: 'perspectives',
    label: 'Gathering perspectives',
    description: 'Collecting independent responses from council models.',
  },
  {
    key: 'ranking',
    label: 'Peer review and ranking',
    description: 'Comparing responses and extracting rankings.',
  },
  {
    key: 'synthesis',
    label: 'Final synthesis',
    description: 'Chairman composes the final answer.',
  },
];

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

function shortModelName(model) {
  if (typeof model !== 'string' || model.length === 0) {
    return 'unknown';
  }
  return model.split('/')[1] || model;
}

function getLatestExchange(messages = []) {
  let assistantIndex = -1;
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    if (messages[i]?.role === 'assistant') {
      assistantIndex = i;
      break;
    }
  }

  if (assistantIndex === -1) {
    let latestUser = null;
    for (let i = messages.length - 1; i >= 0; i -= 1) {
      if (messages[i]?.role === 'user') {
        latestUser = messages[i];
        break;
      }
    }
    return { latestUser, latestAssistant: null };
  }

  const latestAssistant = messages[assistantIndex];

  let latestUser = null;
  for (let i = assistantIndex - 1; i >= 0; i -= 1) {
    if (messages[i]?.role === 'user') {
      latestUser = messages[i];
      break;
    }
  }

  if (!latestUser) {
    for (let i = messages.length - 1; i >= 0; i -= 1) {
      if (messages[i]?.role === 'user') {
        latestUser = messages[i];
        break;
      }
    }
  }

  return { latestUser, latestAssistant };
}

function deriveStatusTimeline(assistantMessage) {
  const hasStage1 = assistantMessage?.stage1 !== null && assistantMessage?.stage1 !== undefined;
  const hasStage2 = assistantMessage?.stage2 !== null && assistantMessage?.stage2 !== undefined;
  const hasStage3 = assistantMessage?.stage3 !== null && assistantMessage?.stage3 !== undefined;

  const loadingStage1 = Boolean(assistantMessage?.loading?.stage1);
  const loadingStage2 = Boolean(assistantMessage?.loading?.stage2);
  const loadingStage3 = Boolean(assistantMessage?.loading?.stage3);

  const stage1Failures = assistantMessage?.metadata?.failures?.stage1?.length || 0;
  const stage2Failures = assistantMessage?.metadata?.failures?.stage2?.length || 0;
  const stage3Failures = assistantMessage?.metadata?.failures?.stage3?.length || 0;

  const stage1Failed = hasStage1 && !loadingStage1 && Array.isArray(assistantMessage?.stage1)
    && assistantMessage.stage1.length === 0 && stage1Failures > 0;
  const stage2Failed = hasStage2 && !loadingStage2 && Array.isArray(assistantMessage?.stage2)
    && assistantMessage.stage2.length === 0 && stage2Failures > 0;
  const stage3Model = typeof assistantMessage?.stage3?.model === 'string'
    ? assistantMessage.stage3.model.toLowerCase()
    : '';
  const stage3Response = typeof assistantMessage?.stage3?.response === 'string'
    ? assistantMessage.stage3.response.toLowerCase()
    : '';
  const stage3Failed = hasStage3 && !loadingStage3 && (
    stage3Failures > 0
    || stage3Model === 'error'
    || stage3Response.includes('all models failed to respond')
    || stage3Response.startsWith('error:')
  );

  if (hasStage3 && !loadingStage3 && !stage1Failed && !stage2Failed && !stage3Failed) {
    return FRIENDLY_STATUS_STEPS.map((step) => ({
      ...step,
      status: 'complete',
    }));
  }

  let activeIndex = 0;

  if (loadingStage3) {
    activeIndex = 4;
  } else if (hasStage3) {
    activeIndex = 4;
  } else if (hasStage2) {
    activeIndex = 4;
  } else if (loadingStage2) {
    activeIndex = 3;
  } else if (hasStage1 && !loadingStage1) {
    activeIndex = 3;
  } else if (loadingStage1) {
    activeIndex = 2;
  } else if (assistantMessage) {
    activeIndex = 1;
  }

  const timeline = FRIENDLY_STATUS_STEPS.map((step, index) => ({
    ...step,
    status: index < activeIndex ? 'complete' : index === activeIndex ? 'active' : 'upcoming',
  }));

  if (stage1Failed) {
    timeline[2] = {
      ...timeline[2],
      status: 'failed',
      description: `No successful council responses (${stage1Failures} failed).`,
    };

    if (!hasStage2) {
      timeline[3] = {
        ...timeline[3],
        status: 'upcoming',
        description: 'Skipped because Stage 1 had no successful responses.',
      };
    }
  }

  if (stage2Failed) {
    timeline[3] = {
      ...timeline[3],
      status: 'failed',
      description: `No successful peer rankings (${stage2Failures} failed).`,
    };
  }

  if (stage3Failed) {
    timeline[4] = {
      ...timeline[4],
      status: 'failed',
      description: `Final synthesis failed (${stage3Failures} failure event${stage3Failures === 1 ? '' : 's'}).`,
    };
  }

  return timeline;
}

function summarizeProgress(steps) {
  const completeCount = steps.filter((step) => step.status === 'complete').length;
  const activeCount = steps.filter((step) => step.status === 'active').length;
  const failedCount = steps.filter((step) => step.status === 'failed').length;
  const doneCount = completeCount;
  const totalCount = steps.length;
  const percent = totalCount > 0 ? Math.round((doneCount / totalCount) * 100) : 0;

  return {
    doneCount,
    completeCount,
    activeCount,
    failedCount,
    totalCount,
    percent,
  };
}

function getActiveStepLabel(steps) {
  const activeStep = steps.find((step) => step.status === 'active');
  if (activeStep) {
    return activeStep.label;
  }
  return steps[steps.length - 1]?.label || 'Final synthesis';
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
  viewMode = VIEW_MODE_STAGES,
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

  const renderStageMode = () => (
    <>
      {conversation.messages.map((msg, index) => (
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
      ))}

      {isLoading && (
        <div className="loading-indicator">
          <div className="spinner"></div>
          <span>Consulting the council...</span>
        </div>
      )}
    </>
  );

  const renderStatusMode = () => {
    const { latestUser, latestAssistant } = getLatestExchange(conversation.messages);
    const statusSteps = deriveStatusTimeline(latestAssistant);
    const { doneCount, failedCount, totalCount, percent } = summarizeProgress(statusSteps);
    const activeStepLabel = getActiveStepLabel(statusSteps);

    const finalResponse = latestAssistant?.stage3;
    const chairmanModel = shortModelName(
      finalResponse?.actual_model || finalResponse?.model || finalResponse?.requested_model
    );

    return (
      <div className="status-mode-layout">
        <div className="status-main-panel">
          <div className="status-card status-question-card">
            <div className="message-label">You</div>
            {latestUser ? (
              <div className="message-content">
                <div className="markdown-content">
                  <MarkdownRenderer content={latestUser.content} />
                </div>
              </div>
            ) : (
              <div className="status-placeholder-text">Waiting for a question...</div>
            )}
          </div>

          <div className="status-card status-chair-card">
            <div className="status-card-header">
              <h3>Chairman Response</h3>
            </div>

            {finalResponse ? (
              <div className="status-chairman-response">
                <div className="chairman-label">Chairman: {chairmanModel}</div>
                <div className="final-text markdown-content">
                  <MarkdownRenderer content={finalResponse.response} />
                </div>
              </div>
            ) : (
              <div className="stage-loading status-waiting-state">
                <div className="spinner"></div>
                <span>{activeStepLabel}...</span>
              </div>
            )}
          </div>

          <ModelAvailability metadata={latestAssistant?.metadata} />
        </div>

        <aside className="status-progress-panel" aria-label="Progress timeline">
          <div className="status-progress-header">
            <h3>Progress</h3>
            <span className={`status-progress-count ${failedCount > 0 ? 'is-failed' : ''}`}>
              {doneCount}/{totalCount}
            </span>
          </div>

          <div className="status-progress-track" aria-hidden="true">
            <div
              className={`status-progress-fill ${failedCount > 0 ? 'status-progress-fill-failed' : ''}`}
              style={{ width: `${percent}%` }}
            ></div>
          </div>

          <ul className="status-step-list">
            {statusSteps.map((step) => (
              <li key={step.key} className={`status-step-item status-${step.status}`}>
                <span className="status-step-marker" aria-hidden="true">
                  {step.status === 'complete' ? 'OK' : step.status === 'active' ? 'IN' : step.status === 'failed' ? '!!' : '..'}
                </span>
                <div className="status-step-text">
                  <div className="status-step-title">{step.label}</div>
                  <div className="status-step-description">{step.description}</div>
                </div>
              </li>
            ))}
          </ul>
        </aside>
      </div>
    );
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
          viewMode === VIEW_MODE_STAGES ? renderStageMode() : renderStatusMode()
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

import { useState } from 'react';
import MarkdownRenderer from './MarkdownRenderer';
import './Stage1.css';

export default function Stage1({ responses, expectedCount }) {
  const [activeTab, setActiveTab] = useState(0);

  if (!responses || responses.length === 0) {
    return null;
  }

  const coverageText =
    typeof expectedCount === 'number' && expectedCount > 0
      ? `Received ${responses.length} of ${expectedCount} model responses`
      : `Received ${responses.length} model response${responses.length === 1 ? '' : 's'}`;

  return (
    <div className="stage stage1">
      <h3 className="stage-title">Stage 1: Individual Responses</h3>
      <p className="stage-meta">{coverageText}</p>

      <div className="tabs">
        {responses.map((resp, index) => (
          <button
            key={index}
            className={`tab ${activeTab === index ? 'active' : ''}`}
            onClick={() => setActiveTab(index)}
          >
            {resp.model.split('/')[1] || resp.model}
          </button>
        ))}
      </div>

      <div className="tab-content">
        <div className="model-name">{responses[activeTab].model}</div>
        <div className="response-text markdown-content">
          <MarkdownRenderer content={responses[activeTab].response} />
        </div>
      </div>
    </div>
  );
}

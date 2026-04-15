import MarkdownRenderer from './MarkdownRenderer';
import './Stage3.css';

export default function Stage3({ finalResponse }) {
  if (!finalResponse) {
    return null;
  }

  const model =
    typeof finalResponse.model === 'string' && finalResponse.model.length > 0
      ? finalResponse.model
      : 'unknown';
  const modelName = model.split('/')[1] || model;

  return (
    <div className="stage stage3">
      <h3 className="stage-title">Stage 3: Final Council Answer</h3>
      <div className="final-response">
        <div className="chairman-label">
          Chairman: {modelName}
        </div>
        <div className="final-text markdown-content">
          <MarkdownRenderer content={finalResponse.response} />
        </div>
      </div>
    </div>
  );
}

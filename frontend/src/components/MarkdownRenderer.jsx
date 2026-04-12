import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import rehypeSanitize from 'rehype-sanitize';

function normalizeModelMarkdown(text) {
  if (!text) return '';

  return text
    .replace(/\r\n/g, '\n')
    .replace(/<br\s*\/?\s*>/gi, '<br />');
}

export default function MarkdownRenderer({ content }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      rehypePlugins={[rehypeRaw, rehypeSanitize]}
    >
      {normalizeModelMarkdown(content)}
    </ReactMarkdown>
  );
}

// src/components/InfoModal.js
import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';

import { markdownContent } from './info';

const InfoModal = ({ isOpen, onClose }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center z-50">
      <div className="bg-white p-6 rounded shadow-lg max-w-3xl w-full relative overflow-y-auto max-h-full">
        <button onClick={onClose} className="text-red-500 absolute top-2 right-2">Close</button>
        <ReactMarkdown
          children={markdownContent}
          remarkPlugins={[remarkMath]}
          rehypePlugins={[rehypeKatex]}
        />
      </div>
    </div>
  );
};

export default InfoModal;
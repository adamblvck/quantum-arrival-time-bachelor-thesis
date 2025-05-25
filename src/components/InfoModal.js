// src/components/InfoModal.js
import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';

import { markdownContent } from './info';

const InfoModal = ({ isOpen, onClose, isDark }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center z-50">
      <div className={`${isDark ? 'bg-gray-800' : 'bg-white'} p-6 rounded shadow-lg max-w-3xl w-full relative overflow-y-auto max-h-full`}>
        <button 
          onClick={onClose} 
          className={`absolute top-2 right-2 px-3 py-1 rounded ${
            isDark 
              ? 'bg-red-600 hover:bg-red-700 text-gray-200' 
              : 'bg-red-500 hover:bg-red-600 text-white'
          }`}
        >
          Close
        </button>
        <ReactMarkdown
          children={markdownContent}
          remarkPlugins={[remarkMath]}
          rehypePlugins={[rehypeKatex]}
          className={`prose ${isDark ? 'prose-invert' : ''} max-w-none`}
          components={{
            h1: ({node, ...props}) => <h1 className={`text-3xl font-bold my-4 ${isDark ? 'text-white' : 'text-gray-900'}`} {...props} />,
            h2: ({node, ...props}) => <h2 className={`text-2xl font-bold my-3 ${isDark ? 'text-white' : 'text-gray-900'}`} {...props} />,
            h3: ({node, ...props}) => <h3 className={`text-xl font-bold my-2 ${isDark ? 'text-white' : 'text-gray-900'}`} {...props} />,
            p: ({node, ...props}) => <p className={`my-2 ${isDark ? 'text-gray-300' : 'text-gray-700'}`} {...props} />,
            a: ({node, ...props}) => <a className={`text-blue-500 hover:text-blue-600 ${isDark ? 'text-blue-400 hover:text-blue-300' : ''}`} {...props} />,
            code: ({node, ...props}) => <code className={`px-1 py-0.5 rounded ${isDark ? 'bg-gray-700 text-gray-200' : 'bg-gray-100 text-gray-800'}`} {...props} />,
          }}
        />
      </div>
    </div>
  );
};

export default InfoModal;
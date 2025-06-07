import React from 'react'

export default function SuggestionsCard({ suggestion }) {
  const { improved_resume, method, confidence } = suggestion

  const getConfidenceColor = (conf) => {
    switch (conf) {
      case 'high': return 'text-green-400'
      case 'medium': return 'text-yellow-400'
      case 'low': return 'text-red-400'
      default: return 'text-gray-400'
    }
  }

  return (
    <div className="bg-gray-900 text-gray-100 p-6 rounded shadow mt-6">
      <div className="flex justify-between items-center mb-4">
        <div className="text-lg font-bold">Resume Suggestions</div>
        <div className="flex items-center gap-2">
          <span className="text-sm text-gray-400">Method:</span>
          <span className="text-sm font-semibold text-blue-400">{method.toUpperCase()}</span>
          <span className="text-sm text-gray-400">Confidence:</span>
          <span className={`text-sm font-semibold ${getConfidenceColor(confidence)}`}>
            {confidence.toUpperCase()}
          </span>
        </div>
      </div>
      <div className="bg-gray-800 p-4 rounded">
        <pre className="whitespace-pre-wrap text-gray-200 font-mono text-sm">{improved_resume}</pre>
      </div>
    </div>
  )
} 
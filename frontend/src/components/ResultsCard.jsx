import React from 'react'

export default function ResultsCard({ ats, match, strengths, weaknesses, improve }) {
  return (
    <div className="bg-gray-900 text-gray-100 p-6 rounded shadow mt-6">
      <div className="flex gap-8 mb-4">
        <div>
          <div className="text-lg font-bold">ATS Score</div>
          <div className="text-2xl text-green-400">{ats}%</div>
        </div>
        <div>
          <div className="text-lg font-bold">Job Match</div>
          <div className="text-2xl text-blue-400">{match}%</div>
        </div>
      </div>
      <div className="mb-2">
        <span className="font-semibold text-green-300">Strengths:</span> {strengths?.join(', ') || '-'}
      </div>
      <div className="mb-2">
        <span className="font-semibold text-red-300">Weaknesses:</span> {weaknesses?.join(', ') || '-'}
      </div>
      <div>
        <span className="font-semibold text-yellow-300">Areas to Improve:</span> {improve?.join(', ') || '-'}
      </div>
    </div>
  )
} 
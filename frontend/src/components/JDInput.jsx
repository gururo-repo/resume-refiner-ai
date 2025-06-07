import React from 'react'

export default function JDInput({ value, onChange }) {
  return (
    <div className="bg-gray-800 p-4 rounded shadow mt-4">
      <label className="block text-gray-200 mb-2">Paste Job Description</label>
      <textarea value={value} onChange={e => onChange(e.target.value)} rows={6} className="w-full bg-gray-700 text-gray-100 rounded border-none p-2" />
    </div>
  )
} 
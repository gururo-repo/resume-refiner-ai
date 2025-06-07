import React, { useRef } from 'react'

export default function ResumeUpload({ onUpload }) {
  const fileInput = useRef()
  const handleFile = e => {
    if (e.target.files[0]) onUpload(e.target.files[0])
  }
  return (
    <div className="bg-gray-800 p-4 rounded shadow">
      <label className="block text-gray-200 mb-2">Upload Resume (.pdf or .txt)</label>
      <input type="file" accept=".pdf,.txt" ref={fileInput} onChange={handleFile} className="block w-full text-gray-100 bg-gray-700 border-none rounded" />
    </div>
  )
} 
import React, { useState } from 'react'
import ResumeUpload from './components/ResumeUpload'
import JDInput from './components/JDInput'
import ResultsCard from './components/ResultsCard'
import SuggestionsCard from './components/SuggestionsCard'
import { uploadResume, analyze, genaiSuggest } from './api'

export default function App() {
  const [resumeFile, setResumeFile] = useState(null)
  const [resumeText, setResumeText] = useState('')
  const [jdText, setJDText] = useState('')
  const [results, setResults] = useState(null)
  const [suggestion, setSuggestion] = useState('')
  const [loading, setLoading] = useState(false)

  const handleResumeUpload = async file => {
    setResumeFile(file)
    setLoading(true)
    const { data } = await uploadResume(file)
    setResumeText(data.resume_text)
    setLoading(false)
  }

  const handleAnalyze = async () => {
    setLoading(true)
    const { data } = await analyze(resumeText, jdText)
    setResults(data)
    setLoading(false)
  }

  const handleSuggest = async () => {
    setLoading(true)
    const { data } = await genaiSuggest(resumeText, jdText)
    setSuggestion(data.suggested_resume)
    setLoading(false)
  }

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 flex flex-col items-center py-10">
      <div className="w-full max-w-2xl">
        <h1 className="text-3xl font-bold mb-6 text-center text-blue-300">Resume Refiner & Career Recommender</h1>
        <ResumeUpload onUpload={handleResumeUpload} />
        <JDInput value={jdText} onChange={setJDText} />
        <div className="flex gap-4 mt-4">
          <button onClick={handleAnalyze} disabled={!resumeText || !jdText || loading} className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded disabled:opacity-50">Analyze</button>
          <button onClick={handleSuggest} disabled={!resumeText || !jdText || loading} className="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded disabled:opacity-50">GenAI Suggest</button>
        </div>
        {loading && <div className="mt-4 text-blue-400">Processing...</div>}
        {results && <ResultsCard ats={results.ats_score} match={results.match_pct} strengths={results.strengths} weaknesses={results.weaknesses} improve={results.areas_to_improve} />}
        {suggestion && <SuggestionsCard suggestion={suggestion} />}
      </div>
    </div>
  )
}

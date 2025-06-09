import React, { useState } from 'react';
import { Upload, FileText, Brain, Target, TrendingUp, Loader2, AlertCircle, CheckCircle } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { analyzeResume } from '../api';

const AnalyzerPage = () => {
  const [resumeFile, setResumeFile] = useState(null);
  const [jobDescription, setJobDescription] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState('');

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setResumeFile(file);
      setError('');
    }
  };

  const handleAnalyze = async () => {
    if (!resumeFile || !jobDescription.trim()) {
      setError('Please provide both a resume file and job description');
      return;
    }

    setIsAnalyzing(true);
    setError('');
    setResults(null);

    try {
      console.log('Starting analysis...');
      const response = await analyzeResume(resumeFile, jobDescription);
      console.log('Analysis complete:', response.data);
      setResults(response.data);
    } catch (err) {
      console.error('Analysis error:', err);
      setError(err.response?.data?.error || 'Failed to analyze resume. Please try again.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const ScoreCard = ({ title, score, color }) => (
    <div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 text-center border border-cyan-500/10 hover:border-cyan-500/20 transition-all">
      <h3 className="text-lg font-medium text-cyan-100 mb-3">{title}</h3>
      <div className="text-4xl font-bold" style={{ color }}>
        {typeof score === 'number' ? score.toFixed(1) : '0.0'}%
      </div>
    </div>
  );

  const AnalysisSection = ({ title, items, icon: Icon, color }) => (
    <div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-cyan-500/10">
      <h2 className="text-xl font-semibold mb-4 flex items-center gap-2 text-cyan-50">
        <Icon className={`h-5 w-5 ${color}`} />
        {title}
      </h2>
      <ul className="space-y-3">
        {items.map((item, index) => (
          <li key={index} className="flex items-start gap-3 text-cyan-100">
            <Icon className={`h-5 w-5 mt-1 flex-shrink-0 ${color}`} />
            <span>{item}</span>
          </li>
        ))}
      </ul>
    </div>
  );

  const isFallbackMode = results?.analysis_source === 'fallback';

  return (
    <div className="min-h-screen bg-[#0A0A0A] text-white pt-20">
      <div className="max-w-7xl mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold mb-4 text-cyan-50">
            Resume Analyzer
          </h1>
          <p className="text-cyan-200/70 text-lg">
            Get AI-powered insights for your resume
          </p>
        </div>

        {/* Input Section */}
        <div className="grid md:grid-cols-2 gap-8 mb-12">
          {/* Resume Upload */}
          <Card className="bg-white/5 backdrop-blur-sm border-cyan-500/10">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-cyan-50">
                <Upload className="h-5 w-5 text-cyan-400" />
                Resume Upload
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="border-2 border-dashed border-cyan-500/20 rounded-xl p-8 text-center hover:border-cyan-400/50 transition-all">
                <input
                  type="file"
                  accept=".pdf,.doc,.docx"
                  onChange={handleFileUpload}
                  className="hidden"
                  id="resume-upload"
                />
                <label htmlFor="resume-upload" className="cursor-pointer">
                  <FileText className="h-12 w-12 text-cyan-400 mx-auto mb-4" />
                  <p className="text-cyan-100 mb-2">
                    {resumeFile ? resumeFile.name : 'Click to upload resume'}
                  </p>
                  <p className="text-sm text-cyan-200/50">
                    Supports PDF, DOC, DOCX
                  </p>
                </label>
              </div>
            </CardContent>
          </Card>

          {/* Job Description */}
          <Card className="bg-white/5 backdrop-blur-sm border-cyan-500/10">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-cyan-50">
                <FileText className="h-5 w-5 text-cyan-400" />
                Job Description
              </CardTitle>
            </CardHeader>
            <CardContent>
              <textarea
                value={jobDescription}
                onChange={(e) => setJobDescription(e.target.value)}
                placeholder="Paste the job description here..."
                className="w-full h-64 p-4 bg-white/5 border border-cyan-500/20 rounded-xl text-cyan-100 placeholder-cyan-200/30 focus:border-cyan-400 focus:outline-none transition-all"
              />
            </CardContent>
          </Card>
        </div>

        {/* Error Message */}
        {error && (
          <div className="mb-8 p-4 bg-red-500/10 border border-red-500/50 rounded-xl flex items-center gap-3">
            <AlertCircle className="h-5 w-5 text-red-400" />
            <span className="text-red-300">{error}</span>
          </div>
        )}

        {/* Analyze Button */}
        <div className="text-center mb-12">
          <button
            onClick={handleAnalyze}
            disabled={isAnalyzing || !resumeFile || !jobDescription.trim()}
            className="px-8 py-4 bg-cyan-500 rounded-xl text-white font-medium text-lg hover:bg-cyan-600 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 mx-auto"
          >
            {isAnalyzing ? (
              <>
                <Loader2 className="h-5 w-5 animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                <Brain className="h-5 w-5" />
                Analyze Resume
              </>
            )}
          </button>
        </div>

        {/* Results Section */}
        {results && (
          <div className="space-y-8">
            {/* Fallback Mode Notice */}
            {isFallbackMode && (
              <div className="mb-8 p-4 bg-yellow-500/10 border border-yellow-500/50 rounded-xl flex items-center gap-3">
                <AlertCircle className="h-5 w-5 text-yellow-400" />
                <span className="text-yellow-300">
                  Advanced analysis is currently unavailable.
                </span>
              </div>
            )}

            {/* Scores Overview */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
              <ScoreCard
                title="ATS Score"
                score={results.ats_score}
                color="#3b82f6"
              />
              <ScoreCard
                title="Job Match"
                score={results.job_match_score}
                color="#10b981"
              />
              <ScoreCard
                title="Format Score"
                score={results.format_analysis?.score || 0}
                color="#8b5cf6"
              />
            </div>

            {/* Main Analysis Sections - Only show if not in fallback mode */}
            {!isFallbackMode && (
              <>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                  <AnalysisSection
                    title="Strengths"
                    items={results.strengths}
                    icon={CheckCircle}
                    color="text-green-400"
                  />
                  <AnalysisSection
                    title="Areas for Improvement"
                    items={results.weaknesses}
                    icon={AlertCircle}
                    color="text-red-400"
                  />
                </div>

                {/* Skills Analysis */}
                <div className="bg-white/5 backdrop-blur-sm rounded-xl p-8 border border-cyan-500/10">
                  <h2 className="text-xl font-semibold mb-6 flex items-center gap-2 text-cyan-50">
                    <Target className="h-5 w-5 text-cyan-400" />
                    Skills Analysis
                  </h2>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                    <div>
                      <h3 className="text-lg font-medium mb-4 text-green-400">Matching Skills</h3>
                      <ul className="space-y-3">
                        {results.skills_analysis?.matching_skills?.map((skill, index) => (
                          <li key={index} className="flex items-center gap-3 text-cyan-100">
                            <CheckCircle className="h-5 w-5 text-green-400 flex-shrink-0" />
                            <span>{skill}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                    <div>
                      <h3 className="text-lg font-medium mb-4 text-red-400">Missing Skills</h3>
                      <ul className="space-y-3">
                        {results.skills_analysis?.missing_skills?.map((skill, index) => (
                          <li key={index} className="flex items-center gap-3 text-cyan-100">
                            <AlertCircle className="h-5 w-5 text-red-400 flex-shrink-0" />
                            <span>{skill}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>
                </div>

                {/* Recommendations */}
                <div className="bg-white/5 backdrop-blur-sm rounded-xl p-8 border border-cyan-500/10">
                  <h2 className="text-xl font-semibold mb-6 flex items-center gap-2 text-cyan-50">
                    <TrendingUp className="h-5 w-5 text-cyan-400" />
                    Recommendations
                  </h2>
                  <ul className="space-y-4">
                    {results.improvements?.map((improvement, index) => (
                      <li key={index} className="flex items-start gap-3 text-cyan-100">
                        <TrendingUp className="h-5 w-5 mt-1 text-cyan-400 flex-shrink-0" />
                        <span>{improvement}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default AnalyzerPage;

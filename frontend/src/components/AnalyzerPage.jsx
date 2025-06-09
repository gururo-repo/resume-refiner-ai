import React, { useState } from 'react';
import { Upload, FileText, Brain, Target, TrendingUp, Download, Loader2, AlertCircle, CheckCircle } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { analyzeResume, generateReport } from '../api';

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

  const handleDownloadReport = async () => {
    if (!results) return;

    try {
      const response = await generateReport(results);
      const blob = new Blob([response.data], { type: 'application/pdf' });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = 'resume_analysis_report.pdf';
      link.click();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to generate report');
      console.error('Report error:', err);
    }
  };

  const ScoreCard = ({ title, score, color }) => (
    <div className="bg-white/10 rounded-lg p-4 text-center">
      <h3 className="text-lg font-semibold mb-2">{title}</h3>
      <div className="text-3xl font-bold" style={{ color }}>
        {score}%
      </div>
    </div>
  );

  const AnalysisSection = ({ title, items, icon: Icon, color }) => (
    <div className="bg-white/10 rounded-lg p-6">
      <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
        <Icon className={`h-6 w-6 ${color}`} />
        {title}
      </h2>
      <ul className="space-y-3">
        {items.map((item, index) => (
          <li key={index} className="flex items-start gap-2">
            <Icon className={`h-5 w-5 mt-1 ${color}`} />
            <span>{item}</span>
          </li>
        ))}
      </ul>
    </div>
  );

  const isFallbackMode = results?.analysis_source === 'fallback';

  return (
    <div className="min-h-screen bg-black text-white p-4">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
            Resume Analyzer
          </h1>
          <p className="text-gray-300 text-lg">
            Upload your resume and job description to get AI-powered insights
          </p>
        </div>

        {/* Input Section */}
        <div className="grid md:grid-cols-2 gap-6 mb-8">
          {/* Resume Upload */}
          <Card className="bg-white/5 border-white/10">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-cyan-400">
                <Upload className="h-5 w-5" />
                Resume Upload
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="border-2 border-dashed border-white/20 rounded-lg p-6 text-center hover:border-cyan-500 transition-colors">
                <input
                  type="file"
                  accept=".pdf,.doc,.docx"
                  onChange={handleFileUpload}
                  className="hidden"
                  id="resume-upload"
                />
                <label htmlFor="resume-upload" className="cursor-pointer">
                  <FileText className="h-12 w-12 text-cyan-400 mx-auto mb-4" />
                  <p className="text-gray-300 mb-2">
                    {resumeFile ? resumeFile.name : 'Click to upload resume'}
                  </p>
                  <p className="text-sm text-gray-500">
                    Supports PDF, DOC, DOCX
                  </p>
                </label>
              </div>
            </CardContent>
          </Card>

          {/* Job Description */}
          <Card className="bg-white/5 border-white/10">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-cyan-400">
                <FileText className="h-5 w-5" />
                Job Description
              </CardTitle>
            </CardHeader>
            <CardContent>
              <textarea
                value={jobDescription}
                onChange={(e) => setJobDescription(e.target.value)}
                placeholder="Paste the job description here..."
                className="w-full h-64 p-4 bg-white/5 border border-white/10 rounded-lg text-white placeholder-gray-400 focus:border-cyan-500 focus:outline-none"
              />
            </CardContent>
          </Card>
        </div>

        {/* Error Message */}
        {error && (
          <div className="mb-6 p-4 bg-red-500/20 border border-red-500 rounded-lg flex items-center gap-2">
            <AlertCircle className="h-5 w-5 text-red-400" />
            <span className="text-red-300">{error}</span>
          </div>
        )}

        {/* Analyze Button */}
        <div className="text-center mb-8">
          <button
            onClick={handleAnalyze}
            disabled={isAnalyzing || !resumeFile || !jobDescription.trim()}
            className="px-8 py-4 bg-gradient-to-r from-cyan-500 to-blue-500 rounded-lg text-white font-semibold text-lg hover:from-cyan-600 hover:to-blue-600 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 mx-auto"
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
          <div className="space-y-6">
            {/* Fallback Mode Notice */}
            {isFallbackMode && (
              <div className="mb-6 p-4 bg-yellow-500/20 border border-yellow-500 rounded-lg flex items-center gap-2">
                <AlertCircle className="h-5 w-5 text-yellow-400" />
                <span className="text-yellow-300">
                  Advanced analysis is currently unavailable. Showing basic scores only.
                </span>
              </div>
            )}

            {/* Scores Overview */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
              <ScoreCard
                title="ATS Score"
                score={results.ats_score}
                color="#22d3ee"
              />
              <ScoreCard
                title="Job Match"
                score={results.job_match_score}
                color="#3b82f6"
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
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
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
                <div className="bg-white/10 rounded-lg p-6">
                  <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
                    <Target className="h-6 w-6 text-cyan-400" />
                    Skills Analysis
                  </h2>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <h3 className="text-lg font-semibold mb-2 text-green-400">Matching Skills</h3>
                      <ul className="space-y-2">
                        {results.skills_analysis?.matching_skills?.map((skill, index) => (
                          <li key={index} className="flex items-center gap-2">
                            <CheckCircle className="h-5 w-5 text-green-400" />
                            <span>{skill}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                    <div>
                      <h3 className="text-lg font-semibold mb-2 text-red-400">Missing Skills</h3>
                      <ul className="space-y-2">
                        {results.skills_analysis?.missing_skills?.map((skill, index) => (
                          <li key={index} className="flex items-center gap-2">
                            <AlertCircle className="h-5 w-5 text-red-400" />
                            <span>{skill}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>
                </div>

                {/* Recommendations */}
                <div className="bg-white/10 rounded-lg p-6">
                  <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
                    <TrendingUp className="h-6 w-6 text-blue-400" />
                    Recommendations
                  </h2>
                  <ul className="space-y-3">
                    {results.improvements?.map((improvement, index) => (
                      <li key={index} className="flex items-start gap-2">
                        <TrendingUp className="h-5 w-5 mt-1 text-blue-400" />
                        <span>{improvement}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                {/* Download Report Button */}
                <div className="text-center">
                  <button
                    onClick={handleDownloadReport}
                    className="px-6 py-3 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg text-white font-semibold hover:from-purple-600 hover:to-pink-600 transition-all flex items-center gap-2 mx-auto"
                  >
                    <Download className="h-5 w-5" />
                    Download Full Report
                  </button>
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

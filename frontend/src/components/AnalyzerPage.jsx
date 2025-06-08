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

    try {
      const response = await analyzeResume(resumeFile, jobDescription);
      setResults(response.data);
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to analyze resume. Please try again.');
      console.error('Analysis error:', err);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleDownloadReport = async () => {
    if (!results) return;

    try {
      const response = await generateReport(results);
      
      // Create blob from response
      const blob = new Blob([response.data], { type: 'application/pdf' });
      const url = window.URL.createObjectURL(blob);
      
      // Download PDF
      const link = document.createElement('a');
      link.href = url;
      link.download = 'resume_analysis_report.pdf';
      link.click();
      
      // Cleanup
      window.URL.revokeObjectURL(url);
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to generate report');
      console.error('Report error:', err);
    }
  };

  const getScoreColor = (score) => {
    if (score >= 80) return 'text-green-400';
    if (score >= 60) return 'text-yellow-400';
    return 'text-red-400';
  };

  const getScoreStatus = (score) => {
    if (score >= 80) return 'Excellent';
    if (score >= 60) return 'Good';
    if (score >= 40) return 'Fair';
    return 'Needs Improvement';
  };

  return (
    <div className="min-h-screen bg-black text-white p-4">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold gradient-title mb-4">Resume Analyzer</h1>
          <p className="text-gray-300 text-lg">
            Upload your resume and job description to get AI-powered insights
          </p>
        </div>

        {/* Input Section */}
        <div className="grid md:grid-cols-2 gap-6 mb-8">
          {/* Resume Upload */}
          <Card className="glass-card">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-cyan-100">
                <Upload className="h-5 w-5" />
                Resume Upload
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="border-2 border-dashed border-gray-600 rounded-lg p-6 text-center hover:border-cyan-500 transition-colors">
                <input
                  type="file"
                  accept=".pdf,.doc,.docx,.txt"
                  onChange={handleFileUpload}
                  className="hidden"
                  id="resume-upload"
                />
                <label htmlFor="resume-upload" className="cursor-pointer">
                  <FileText className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                  <p className="text-gray-300 mb-2">
                    {resumeFile ? resumeFile.name : 'Click to upload resume'}
                  </p>
                  <p className="text-sm text-gray-500">
                    Supports PDF, DOC, DOCX, TXT
                  </p>
                </label>
              </div>
            </CardContent>
          </Card>

          {/* Job Description */}
          <Card className="glass-card">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-cyan-100">
                <FileText className="h-5 w-5" />
                Job Description
              </CardTitle>
            </CardHeader>
            <CardContent>
              <textarea
                value={jobDescription}
                onChange={(e) => setJobDescription(e.target.value)}
                placeholder="Paste the job description here..."
                className="w-full h-64 p-4 bg-gray-800 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:border-cyan-500 focus:outline-none"
              />
            </CardContent>
          </Card>
        </div>

        {/* Error Message */}
        {error && (
          <div className="mb-6 p-4 bg-red-900/50 border border-red-500 rounded-lg flex items-center gap-2">
            <AlertCircle className="h-5 w-5 text-red-400" />
            <span className="text-red-300">{error}</span>
          </div>
        )}

        {/* Analyze Button */}
        <div className="text-center mb-8">
          <button
            onClick={handleAnalyze}
            disabled={isAnalyzing || !resumeFile || !jobDescription.trim()}
            className="btn-primary text-lg px-8 py-4 disabled:opacity-50 disabled:cursor-not-allowed"
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
            {/* Score Overview */}
            <div className="grid md:grid-cols-3 gap-6">
              <Card className="glass-card text-center">
                <CardContent className="pt-6">
                  <Target className="h-12 w-12 text-cyan-400 mx-auto mb-4" />
                  <h3 className="text-lg font-semibold mb-2">Match Score</h3>
                  <div className={`text-3xl font-bold ${getScoreColor(results.match_score)}`}>
                    {results.match_score}%
                  </div>
                  <p className="text-sm text-gray-400 mt-2">
                    {getScoreStatus(results.match_score)}
                  </p>
                </CardContent>
              </Card>

              <Card className="glass-card text-center">
                <CardContent className="pt-6">
                  <TrendingUp className="h-12 w-12 text-cyan-400 mx-auto mb-4" />
                  <h3 className="text-lg font-semibold mb-2">Role Match</h3>
                  <div className="text-3xl font-bold text-cyan-400">
                    {results.role_prediction?.category || 'Unknown'}
                  </div>
                  <p className="text-sm text-gray-400 mt-2">
                    Confidence: {results.role_prediction?.confidence || 0}%
                  </p>
                </CardContent>
              </Card>

              <Card className="glass-card text-center">
                <CardContent className="pt-6">
                  <Brain className="h-12 w-12 text-cyan-400 mx-auto mb-4" />
                  <h3 className="text-lg font-semibold mb-2">AI Analysis</h3>
                  <button
                    onClick={handleDownloadReport}
                    className="mt-2 px-4 py-2 bg-cyan-600 text-white rounded-lg hover:bg-cyan-700 transition-colors flex items-center gap-2 mx-auto"
                  >
                    <Download className="h-4 w-4" />
                    Download Report
                  </button>
                </CardContent>
              </Card>
            </div>

            {/* Detailed Analysis */}
            <div className="grid md:grid-cols-2 gap-6">
              {/* Resume Data */}
              <Card className="glass-card">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-cyan-400">
                    <FileText className="h-5 w-5" />
                    Resume Analysis
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div>
                      <h4 className="font-semibold text-gray-300 mb-2">Skills</h4>
                      <div className="flex flex-wrap gap-2">
                        {results.resume_data?.skills?.map((skill, index) => (
                          <span key={index} className="px-3 py-1 bg-cyan-900/50 text-cyan-300 rounded-full text-sm">
                            {skill}
                          </span>
                        ))}
                      </div>
                    </div>
                    <div>
                      <h4 className="font-semibold text-gray-300 mb-2">Experience</h4>
                      <p className="text-gray-400">
                        {results.resume_data?.experience?.[0] || 'No experience found'}
                      </p>
                    </div>
                    <div>
                      <h4 className="font-semibold text-gray-300 mb-2">Education</h4>
                      <p className="text-gray-400">
                        {results.resume_data?.education?.[0] || 'No education found'}
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Job Analysis */}
              <Card className="glass-card">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-cyan-400">
                    <Target className="h-5 w-5" />
                    Job Analysis
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div>
                      <h4 className="font-semibold text-gray-300 mb-2">Required Skills</h4>
                      <div className="flex flex-wrap gap-2">
                        {results.job_analysis?.required_skills?.map((skill, index) => (
                          <span key={index} className="px-3 py-1 bg-cyan-900/50 text-cyan-300 rounded-full text-sm">
                            {skill}
                          </span>
                        ))}
                      </div>
                    </div>
                    <div>
                      <h4 className="font-semibold text-gray-300 mb-2">Experience Level</h4>
                      <p className="text-gray-400">
                        {results.job_analysis?.experience_level || 'Not specified'}
                      </p>
                    </div>
                    <div>
                      <h4 className="font-semibold text-gray-300 mb-2">Education Requirements</h4>
                      <p className="text-gray-400">
                        {results.job_analysis?.education_requirements || 'Not specified'}
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Match Components */}
            <Card className="glass-card">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-cyan-400">
                  <TrendingUp className="h-5 w-5" />
                  Match Components
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid md:grid-cols-3 gap-4">
                  {results.match_components && Object.entries(results.match_components).map(([key, value]) => (
                    <div key={key} className="text-center">
                      <h4 className="font-semibold text-gray-300 mb-2 capitalize">
                        {key.replace(/_/g, ' ')}
                      </h4>
                      <div className={`text-2xl font-bold ${getScoreColor(value)}`}>
                        {value}%
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* ATS Score Section */}
            <div className="bg-white rounded-lg shadow-md p-6 mb-6">
                <h2 className="text-2xl font-bold mb-4">ATS Score Analysis</h2>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                        <div className="flex items-center justify-center mb-4">
                            <div className="relative">
                                <svg className="w-32 h-32">
                                    <circle
                                        className="text-gray-200"
                                        strokeWidth="8"
                                        stroke="currentColor"
                                        fill="transparent"
                                        r="56"
                                        cx="64"
                                        cy="64"
                                    />
                                    <circle
                                        className="text-blue-600"
                                        strokeWidth="8"
                                        strokeDasharray={352}
                                        strokeDashoffset={352 - (352 * (results.ats_score?.overall_score || 0)) / 100}
                                        strokeLinecap="round"
                                        stroke="currentColor"
                                        fill="transparent"
                                        r="56"
                                        cx="64"
                                        cy="64"
                                    />
                                </svg>
                                <div className="absolute inset-0 flex items-center justify-center">
                                    <span className="text-2xl font-bold">
                                        {results.ats_score?.overall_score || 0}%
                                    </span>
                                </div>
                            </div>
                        </div>
                        <div className="space-y-2">
                            {results.ats_score?.components && Object.entries(results.ats_score.components).map(([key, value]) => (
                                <div key={key} className="flex justify-between items-center">
                                    <span className="capitalize">{key.replace('_', ' ')}</span>
                                    <span className="font-semibold">{value}%</span>
                                </div>
                            ))}
                        </div>
                    </div>
                    <div className="space-y-4">
                        {results.ats_score?.strengths && results.ats_score.strengths.length > 0 && (
                            <div>
                                <h3 className="text-lg font-semibold text-green-600 mb-2">Strengths</h3>
                                <ul className="list-disc list-inside space-y-1">
                                    {results.ats_score.strengths.map((strength, index) => (
                                        <li key={index} className="text-gray-700">{strength}</li>
                                    ))}
                                </ul>
                            </div>
                        )}
                        {results.ats_score?.weaknesses && results.ats_score.weaknesses.length > 0 && (
                            <div>
                                <h3 className="text-lg font-semibold text-red-600 mb-2">Areas for Improvement</h3>
                                <ul className="list-disc list-inside space-y-1">
                                    {results.ats_score.weaknesses.map((weakness, index) => (
                                        <li key={index} className="text-gray-700">{weakness}</li>
                                    ))}
                                </ul>
                            </div>
                        )}
                        {results.ats_score?.improvements && results.ats_score.improvements.length > 0 && (
                            <div>
                                <h3 className="text-lg font-semibold text-blue-600 mb-2">Recommendations</h3>
                                <ul className="list-disc list-inside space-y-1">
                                    {results.ats_score.improvements.map((improvement, index) => (
                                        <li key={index} className="text-gray-700">{improvement}</li>
                                    ))}
                                </ul>
                            </div>
                        )}
                    </div>
                </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default AnalyzerPage;

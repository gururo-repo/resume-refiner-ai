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
                  <h3 className="text-lg font-semibold mb-2">ATS Score</h3>
                  <div className={`text-3xl font-bold ${getScoreColor(results.ats_score)}`}>
                    {results.ats_score}%
                  </div>
                  <p className="text-sm text-gray-400 mt-2">
                    {getScoreStatus(results.ats_score)}
                  </p>
                </CardContent>
              </Card>

              <Card className="glass-card text-center">
                <CardContent className="pt-6">
                  <TrendingUp className="h-12 w-12 text-cyan-400 mx-auto mb-4" />
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
              {/* Strengths */}
              <Card className="glass-card">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-green-400">
                    <CheckCircle className="h-5 w-5" />
                    Strengths
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <ul className="space-y-2">
                    {results.strengths.map((strength, index) => (
                      <li key={index} className="flex items-start gap-2">
                        <CheckCircle className="h-5 w-5 text-green-400 mt-1 flex-shrink-0" />
                        <span>{strength}</span>
                      </li>
                    ))}
                  </ul>
                </CardContent>
              </Card>

              {/* Areas for Improvement */}
              <Card className="glass-card">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-red-400">
                    <AlertCircle className="h-5 w-5" />
                    Areas for Improvement
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <ul className="space-y-2">
                    {results.weaknesses.map((weakness, index) => (
                      <li key={index} className="flex items-start gap-2">
                        <AlertCircle className="h-5 w-5 text-red-400 mt-1 flex-shrink-0" />
                        <span>{weakness}</span>
                      </li>
                    ))}
                  </ul>
                </CardContent>
              </Card>
            </div>

            {/* Recommendations */}
            <Card className="glass-card">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-cyan-400">
                  <Brain className="h-5 w-5" />
                  Recommendations
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="space-y-2">
                  {results.recommendations.map((rec, index) => (
                    <li key={index} className="flex items-start gap-2">
                      <Brain className="h-5 w-5 text-cyan-400 mt-1 flex-shrink-0" />
                      <span>{rec}</span>
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>

            {/* Predicted Roles */}
            {results.predicted_roles && results.predicted_roles.length > 0 && (
              <Card className="glass-card">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-purple-400">
                    <Target className="h-5 w-5" />
                    Predicted Roles
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex flex-wrap gap-2">
                    {results.predicted_roles.map((role, index) => (
                      <span
                        key={index}
                        className="px-3 py-1 bg-purple-900/50 text-purple-300 rounded-full text-sm"
                      >
                        {role}
                      </span>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Skill Matches */}
            {results.skill_matches && results.skill_matches.length > 0 && (
              <Card className="glass-card">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-blue-400">
                    <CheckCircle className="h-5 w-5" />
                    Matching Skills
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex flex-wrap gap-2">
                    {results.skill_matches.map((skill, index) => (
                      <span
                        key={index}
                        className="px-3 py-1 bg-blue-900/50 text-blue-300 rounded-full text-sm"
                      >
                        {skill}
                      </span>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Missing Skills */}
            {results.missing_skills && results.missing_skills.length > 0 && (
              <Card className="glass-card">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-orange-400">
                    <AlertCircle className="h-5 w-5" />
                    Missing Skills
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex flex-wrap gap-2">
                    {results.missing_skills.map((skill, index) => (
                      <span
                        key={index}
                        className="px-3 py-1 bg-orange-900/50 text-orange-300 rounded-full text-sm"
                      >
                        {skill}
                      </span>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default AnalyzerPage;

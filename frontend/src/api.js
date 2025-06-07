import axios from 'axios'

const API = axios.create({ baseURL: 'http://localhost:5000' })

export const analyzeResume = (resumeFile, resumeText, jobDescription) => {
  const form = new FormData()
  if (resumeFile) {
    form.append('resume_file', resumeFile)
  } else {
    form.append('resume_text', resumeText)
  }
  form.append('job_description', jobDescription)
  return API.post('/api/analyze-resume', form)
}

export const generateReport = (analysisData) => {
  return API.post('/api/generate-report', analysisData)
}

export const uploadResume = file => {
  const form = new FormData()
  form.append('resume', file)
  return API.post('/upload-resume', form)
}

export const uploadJD = jd_text => API.post('/upload-jd', { jd_text })

export const analyze = (resume_text, jd_text) => API.post('/analyze', { resume_text, jd_text })

export const genaiSuggest = (resume_text, jd_text) => API.post('/genai-suggest', { resume_text, jd_text }) 
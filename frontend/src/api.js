import axios from 'axios'

const API = axios.create({ 
  baseURL: 'http://localhost:5000',
  headers: {
    'Content-Type': 'application/json'
  }
})

export const analyzeResume = (resumeFile, jobDescription) => {
  const form = new FormData()
  form.append('resume', resumeFile)
  form.append('job_description', jobDescription)
  return API.post('/api/analyze', form, {
    headers: {
      'Content-Type': 'multipart/form-data'
    }
  })
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
import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'https://resume-refiner-backend.onrender.com/api'

const API = axios.create({ 
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json'
  },
  withCredentials: true
})

// Add health check function
export const checkHealth = async () => {
  try {
    const response = await API.get('/health')
    return response.data
  } catch (error) {
    console.error('Health check failed:', error)
    throw error
  }
}

export const analyzeResume = async (resumeFile, jobDescription) => {
  try {
    const formData = new FormData()
    formData.append('resume', resumeFile)
    formData.append('job_description', jobDescription)

    console.log('Sending request to analyze resume...')
    const response = await API.post('/analyze', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })

    console.log('Received response:', response.data)
    
    // Ensure we have a valid response structure
    const analysis = {
      ats_score: response.data.ats_score || 0,
      job_match_score: response.data.job_match_score || 0,
      strengths: response.data.strengths || [],
      weaknesses: response.data.weaknesses || [],
      improvements: response.data.improvements || [],
      format_analysis: response.data.format_analysis || {
        score: 0,
        issues: [],
        suggestions: []
      },
      skills_analysis: response.data.skills_analysis || {
        matching_skills: [],
        missing_skills: [],
        skill_gaps: []
      },
      role_prediction: response.data.role_prediction || {
        category: 'Unknown',
        confidence: 0
      },
      job_description_analysis: response.data.job_description_analysis || {}
    }

    return { data: analysis }
  } catch (error) {
    console.error('Error analyzing resume:', error)
    throw error
  }
}

export const analyze = (resume_text, jd_text) => API.post('/analyze', { resume_text, jd_text })

export const uploadResume = file => {
  const form = new FormData()
  form.append('resume', file)
  return API.post('/upload-resume', form)
}

export const uploadJD = jd_text => API.post('/upload-jd', { jd_text })

export const genaiSuggest = (resume_text, jd_text) => API.post('/genai-suggest', { resume_text, jd_text }) 
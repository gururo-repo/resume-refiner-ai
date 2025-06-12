import axios from 'axios'

// Use Vite environment variable or fallback to production URL
const API_BASE_URL = import.meta.env.VITE_API_URL || 'https://resume-refiner-backend.onrender.com/api'

console.log('API Base URL:', API_BASE_URL) // Debug log

const API = axios.create({ 
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json'
  },
  timeout: 30000, // 30 second timeout for file uploads
})

// Add request interceptor for logging
API.interceptors.request.use(
  (config) => {
    console.log(`Making ${config.method?.toUpperCase()} request to: ${config.baseURL}${config.url}`)
    return config
  },
  (error) => {
    console.error('Request interceptor error:', error)
    return Promise.reject(error)
  }
)

// Add response interceptor for better error handling
API.interceptors.response.use(
  (response) => {
    console.log('API Response received:', response.status)
    return response
  },
  (error) => {
    console.error('API Error:', {
      message: error.message,
      status: error.response?.status,
      statusText: error.response?.statusText,
      data: error.response?.data,
      url: error.config?.url
    })
    
    // Handle specific error cases
    if (error.code === 'ECONNABORTED') {
      throw new Error('Request timeout. Please try again.')
    }
    
    if (error.response?.status === 413) {
      throw new Error('File too large. Please upload a smaller file.')
    }
    
    if (error.response?.status >= 500) {
      throw new Error('Server error. Please try again later.')
    }
    
    if (error.response?.status === 404) {
      throw new Error('API endpoint not found. Please check your configuration.')
    }
    
    return Promise.reject(error)
  }
)

export const analyzeResume = async (resumeFile, jobDescription) => {
  try {
    // Validate inputs
    if (!resumeFile || !jobDescription) {
      throw new Error('Both resume file and job description are required')
    }

    // Validate file type
    const allowedTypes = [
      'application/pdf', 
      'application/msword', 
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    ]
    if (!allowedTypes.includes(resumeFile.type)) {
      throw new Error('Please upload a PDF, DOC, or DOCX file')
    }

    // Validate file size (16MB limit)
    const maxSize = 16 * 1024 * 1024 // 16MB
    if (resumeFile.size > maxSize) {
      throw new Error('File size must be less than 16MB')
    }

    const formData = new FormData()
    formData.append('resume', resumeFile)
    formData.append('job_description', jobDescription)

    console.log('Sending request to analyze resume...')
    console.log('File:', resumeFile.name, 'Size:', resumeFile.size, 'Type:', resumeFile.type)
    console.log('Job description length:', jobDescription.length)
    
    const response = await API.post('/analyze', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      },
      timeout: 60000, // 60 seconds for analysis
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
      analysis_source: response.data.analysis_source || 'unknown'
    }

    return { data: analysis }
  } catch (error) {
    console.error('Error analyzing resume:', error)
    
    // Re-throw with more specific error messages
    if (error.response?.data?.error) {
      throw new Error(error.response.data.error)
    }
    
    if (error.message) {
      throw error
    }
    
    throw new Error('Failed to analyze resume. Please try again.')
  }
}

// Legacy functions - keeping for compatibility
export const analyze = (resume_text, jd_text) => API.post('/analyze', { resume_text, jd_text })

export const uploadResume = file => {
  const form = new FormData()
  form.append('resume', file)
  return API.post('/upload-resume', form)
}

export const uploadJD = jd_text => API.post('/upload-jd', { jd_text })

export const genaiSuggest = (resume_text, jd_text) => API.post('/genai-suggest', { resume_text, jd_text })

// Health check function
export const checkHealth = async () => {
  try {
    const response = await API.get('/health')
    console.log('Health check response:', response.data)
    return response
  } catch (error) {
    console.error('Health check failed:', error)
    throw error
  }
}

export default API
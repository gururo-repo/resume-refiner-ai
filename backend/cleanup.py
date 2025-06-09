import os
import shutil
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Essential files and directories to preserve
ESSENTIAL_FILES = {
    'backend/app.py',
    'backend/requirements.txt',
    'backend/utils/model_loader.py',
    'backend/utils/ats_scorer.py',
    'backend/utils/groq_analyzer.py',
    'backend/utils/role_recommender.py',
    'backend/utils/resume_parser.py',
    'backend/utils/skill_extractor.py',
    'backend/utils/format_analyzer.py',
    'backend/utils/content_analyzer.py',
    'backend/utils/improvement_generator.py',
    'backend/utils/report_generator.py',
    'backend/utils/__init__.py',
    'backend/notebooks/resume_evaluator_pipeline.ipynb',
    'backend/data/UpdatedResumeDataSet.csv',
    'backend/data/IT_Job_Roles_Skills.csv',
    'backend/README.md',
    'backend/.env',
    'backend/.gitignore',
    'backend/setup_backend.sh',
    'backend/start_backend.sh',
    'frontend/package.json',
    'frontend/package-lock.json',
    'frontend/src',
    'frontend/index.html',
    'frontend/vite.config.js',
    'frontend/tailwind.config.js',
    'frontend/postcss.config.js',
    'frontend/eslint.config.js',
    'frontend/.gitignore',
    'README.md',
    '.gitignore'
}

def cleanup_directory():
    """Clean up unnecessary files and directories."""
    try:
        # Get the project root directory
        root_dir = Path(__file__).parent.parent

        # Items to remove
        items_to_remove = [
            # Cache and temporary files
            'backend/cache/*',
            'backend/__pycache__',
            'backend/*/__pycache__',
            'backend/*/*/__pycache__',
            'backend/.pytest_cache',
            'backend/.coverage',
            'backend/htmlcov',
            'backend/.mypy_cache',
            'backend/.ruff_cache',
            
            # Temporary model files
            'backend/models/fallback/*',
            'backend/models/*.pkl',
            'backend/models/*.joblib',
            
            # Uploads
            'backend/uploads/*',
            
            # Development files
            'backend/*.log',
            'backend/*.pyc',
            'backend/*.pyo',
            'backend/*.pyd',
            'backend/.Python',
            'backend/build',
            'backend/develop-eggs',
            'backend/dist',
            'backend/downloads',
            'backend/eggs',
            'backend/.eggs',
            'backend/lib',
            'backend/lib64',
            'backend/parts',
            'backend/sdist',
            'backend/var',
            'backend/wheels',
            'backend/*.egg-info',
            'backend/.installed.cfg',
            'backend/*.egg',
            
            # Frontend temporary files
            'frontend/node_modules/.cache',
            'frontend/dist',
            'frontend/.vite',
            'frontend/coverage',
            'frontend/.next',
            'frontend/.nuxt',
            'frontend/.output',
            'frontend/.DS_Store',
            'frontend/*.log'
        ]

        # Create necessary directories
        os.makedirs('backend/cache', exist_ok=True)
        os.makedirs('backend/uploads', exist_ok=True)
        os.makedirs('backend/models/fallback', exist_ok=True)

        # Remove unnecessary files and directories
        for pattern in items_to_remove:
            for path in root_dir.glob(pattern):
                try:
                    if path.is_file():
                        path.unlink()
                        logger.info(f"Removed file: {path}")
                    elif path.is_dir():
                        shutil.rmtree(path)
                        logger.info(f"Removed directory: {path}")
                except Exception as e:
                    logger.warning(f"Failed to remove {path}: {str(e)}")

        logger.info("Cleanup completed successfully")
        
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")
        raise

if __name__ == "__main__":
    cleanup_directory() 
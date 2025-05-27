import React from 'react';
import { Link } from 'react-router-dom';
import { FileText, Menu, X } from 'lucide-react';

const Header = () => {
  const [isMenuOpen, setIsMenuOpen] = React.useState(false);

  const toggleMenu = () => {
    setIsMenuOpen(!isMenuOpen);
  };

  return (
    <header className="fixed top-0 left-0 right-0 z-50 bg-black/80 backdrop-blur-sm border-b border-white/10">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center py-4">
          {/* Logo */}
          <Link to="/" className="flex items-center space-x-2 group">
            <div className="p-2 bg-cyan-500/20 rounded-lg border border-cyan-500/30 group-hover:bg-cyan-500/30 transition-all duration-300">
              <FileText className="h-6 w-6 text-cyan-400" />
            </div>
            <span className="text-xl font-bold text-white">Resume Refiner AI</span>
          </Link>

          {/* Desktop Navigation */}
          <nav className="hidden md:flex items-center space-x-8">
            <Link
              to="/"
              className="text-gray-300 hover:text-cyan-400 transition-colors duration-200"
            >
              Home
            </Link>
            
            <Link
              to="/analyzer"
              className="btn-primary"
            >
              Get Started
            </Link>
          </nav>

          {/* Mobile Menu Button */}
          <button
            onClick={toggleMenu}
            className="md:hidden p-2 text-gray-300 hover:text-cyan-400 transition-colors duration-200"
          >
            {isMenuOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
          </button>
        </div>

        {/* Mobile Navigation */}
        {isMenuOpen && (
          <div className="md:hidden py-4 border-t border-white/10">
            <nav className="flex flex-col space-y-4">
              <Link
                to="/"
                className="text-gray-300 hover:text-cyan-400 transition-colors duration-200 py-2"
                onClick={() => setIsMenuOpen(false)}
              >
                Home
              </Link>
              <Link
                to="/analyzer"
                className="text-gray-300 hover:text-cyan-400 transition-colors duration-200 py-2"
                onClick={() => setIsMenuOpen(false)}
              >
                Analyzer
              </Link>
              <Link
                to="/features"
                className="text-gray-300 hover:text-cyan-400 transition-colors duration-200 py-2"
                onClick={() => setIsMenuOpen(false)}
              >
                Features
              </Link>
              <Link
                to="/analyzer"
                className="btn-primary w-fit"
                onClick={() => setIsMenuOpen(false)}
              >
                Get Started
              </Link>
            </nav>
          </div>
        )}
      </div>
    </header>
  );
};

export default Header;

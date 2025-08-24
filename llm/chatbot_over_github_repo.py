# Enhanced GitHub Repository Chatbot with Auto-Clone
# Requirements: pip install openai gitpython python-dotenv streamlit requests

import os
import json
import tempfile
import shutil
from typing import Dict, List, Any, Optional
from pathlib import Path
import git
from dotenv import load_dotenv
import openai
import streamlit as st
from datetime import datetime
import logging
import re
import requests

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedGitHubChatbot:
    """Enhanced GitHub repository chatbot with auto-clone functionality"""
    
    def __init__(self, repo_input: str, openai_api_key: str):
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.repo = None
        self.repo_path = None
        self.is_temporary = False
        
        # Determine if input is URL or local path
        if self._is_github_url(repo_input):
            self.repo_path = self._clone_repository(repo_input)
            self.is_temporary = True
        else:
            self.repo_path = Path(repo_input)
            if not self.repo_path.exists():
                raise ValueError(f"Repository path does not exist: {repo_input}")
        
        # Initialize repository
        try:
            self.repo = git.Repo(self.repo_path)
            logger.info(f"Repository initialized: {self.repo_path}")
        except git.exc.InvalidGitRepositoryError:
            logger.error(f"Invalid Git repository at {self.repo_path}")
            raise ValueError(f"Invalid Git repository at {self.repo_path}")
    
    def _is_github_url(self, url: str) -> bool:
        """Check if the input is a GitHub URL"""
        github_patterns = [
            r'https://github\.com/[\w\-\.]+/[\w\-\.]+',
            r'git@github\.com:[\w\-\.]+/[\w\-\.]+\.git',
            r'https://github\.com/[\w\-\.]+/[\w\-\.]+\.git'
        ]
        return any(re.match(pattern, url) for pattern in github_patterns)
    
    def _clone_repository(self, url: str) -> Path:
        """Clone repository from GitHub URL"""
        try:
            # Create temporary directory
            temp_dir = tempfile.mkdtemp(prefix="github_chatbot_")
            temp_path = Path(temp_dir)
            
            # Extract repo name from URL
            repo_name = url.split('/')[-1].replace('.git', '')
            clone_path = temp_path / repo_name
            
            logger.info(f"Cloning repository {url} to {clone_path}")
            
            # Clone the repository
            git.Repo.clone_from(url, clone_path)
            
            logger.info(f"Repository cloned successfully to {clone_path}")
            return clone_path
            
        except Exception as e:
            logger.error(f"Error cloning repository: {e}")
            raise ValueError(f"Failed to clone repository {url}: {str(e)}")
    
    def cleanup(self):
        """Clean up temporary files"""
        if self.is_temporary and self.repo_path and self.repo_path.exists():
            try:
                shutil.rmtree(self.repo_path.parent)
                logger.info("Temporary repository cleaned up")
            except Exception as e:
                logger.error(f"Error cleaning up temporary files: {e}")
    
    def get_repo_context(self, query: str) -> str:
        """Get repository context relevant to the query"""
        try:
            # Get basic repo info
            repo_info = self._get_basic_repo_info()
            
            # Get relevant files
            relevant_files = self._find_relevant_files(query)
            
            # Get file contents (limited)
            file_contents = self._get_file_contents(relevant_files[:3])  # Limit to 3 files
            
            # Build context
            context = f"""
Repository Information:
- Name: {repo_info['name']}
- Description: {repo_info['description']}
- Total Files: {len(repo_info['files'])}
- Main Language: {repo_info['main_language']}
- Repository URL: {repo_info.get('remote_url', 'Local repository')}

Recent Commits:
{self._format_recent_commits(repo_info['recent_commits'])}

Repository Structure:
{self._format_file_structure(repo_info['files'][:20])}  # Show first 20 files

Relevant Files for Query:
{self._format_relevant_files(relevant_files[:5])}

File Contents:
{file_contents}
"""
            return context
            
        except Exception as e:
            logger.error(f"Error getting repository context: {e}")
            return f"Error analyzing repository: {str(e)}"
    
    def _get_basic_repo_info(self) -> Dict[str, Any]:
        """Get basic repository information"""
        files = []
        for root, dirs, filenames in os.walk(self.repo_path):
            dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'node_modules', '.venv', 'venv']]
            for filename in filenames:
                if not filename.startswith('.') and not filename.endswith('.pyc'):
                    rel_path = os.path.relpath(os.path.join(root, filename), self.repo_path)
                    files.append(rel_path)
        
        # Get recent commits
        recent_commits = []
        if self.repo:
            try:
                for commit in list(self.repo.iter_commits(max_count=5)):
                    recent_commits.append({
                        'hash': commit.hexsha[:8],
                        'message': commit.message.strip(),
                        'author': str(commit.author),
                        'date': commit.committed_datetime.strftime('%Y-%m-%d %H:%M:%S')
                    })
            except Exception as e:
                logger.error(f"Error getting commits: {e}")
        
        # Get remote URL
        remote_url = None
        if self.repo:
            try:
                remote_url = self.repo.remote().url
            except Exception as e:
                logger.error(f"Error getting remote URL: {e}")
        
        # Determine main language
        main_language = self._detect_main_language(files)
        
        # Get description
        description = self._get_repo_description()
        
        return {
            'name': self.repo_path.name,
            'description': description,
            'files': files,
            'recent_commits': recent_commits,
            'main_language': main_language,
            'remote_url': remote_url
        }
    
    def _find_relevant_files(self, query: str) -> List[str]:
        """Find files relevant to the query"""
        files = []
        for root, dirs, filenames in os.walk(self.repo_path):
            dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'node_modules', '.venv', 'venv']]
            for filename in filenames:
                if not filename.startswith('.') and not filename.endswith('.pyc'):
                    rel_path = os.path.relpath(os.path.join(root, filename), self.repo_path)
                    files.append(rel_path)
        
        # Filter files based on query
        query_lower = query.lower()
        relevant_files = []
        
        for file_path in files:
            # Check if query terms are in file path or name
            if any(term in file_path.lower() for term in query_lower.split()):
                relevant_files.append(file_path)
        
        # If no specific matches, return common important files
        if not relevant_files:
            important_files = [f for f in files if any(
                f.lower().endswith(ext) for ext in ['.py', '.js', '.java', '.cpp', '.c', '.go', '.rs', '.rb', '.md', '.txt', '.json', '.yaml', '.yml']
            ) or any(
                name in f.lower() for name in ['readme', 'main', 'index', 'app', 'server', 'client', 'config', 'setup', 'requirements']
            )]
            relevant_files = important_files[:10]
        
        return relevant_files[:10]  # Limit to 10 files
    
    def _get_file_contents(self, file_paths: List[str]) -> str:
        """Get contents of specified files"""
        contents = []
        for file_path in file_paths:
            try:
                full_path = self.repo_path / file_path
                if full_path.exists() and full_path.stat().st_size < 50000:  # Limit file size to 50KB
                    with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        # Limit content length
                        if len(content) > 3000:
                            content = content[:3000] + "\n... (truncated)"
                        contents.append(f"=== {file_path} ===\n{content}\n")
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}")
        
        return "\n".join(contents)
    
    def _detect_main_language(self, files: List[str]) -> str:
        """Detect the main programming language"""
        extensions = {}
        for file_path in files:
            ext = Path(file_path).suffix.lower()
            if ext in ['.py', '.js', '.java', '.cpp', '.c', '.go', '.rs', '.rb', '.php', '.cs', '.ts', '.jsx', '.tsx']:
                extensions[ext] = extensions.get(ext, 0) + 1
        
        if extensions:
            main_ext = max(extensions.keys(), key=lambda x: extensions[x])
            lang_map = {
                '.py': 'Python', '.js': 'JavaScript', '.java': 'Java',
                '.cpp': 'C++', '.c': 'C', '.go': 'Go', '.rs': 'Rust',
                '.rb': 'Ruby', '.php': 'PHP', '.cs': 'C#', '.ts': 'TypeScript',
                '.jsx': 'React/JSX', '.tsx': 'TypeScript/React'
            }
            return lang_map.get(main_ext, 'Unknown')
        
        return 'Unknown'
    
    def _get_repo_description(self) -> str:
        """Get repository description from README"""
        readme_files = ['README.md', 'README.txt', 'README.rst', 'README', 'readme.md', 'readme.txt']
        for readme in readme_files:
            readme_path = self.repo_path / readme
            if readme_path.exists():
                try:
                    with open(readme_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Get first few lines that contain actual content
                        lines = content.split('\n')
                        description_lines = []
                        for line in lines:
                            line = line.strip()
                            if line and not line.startswith('#'):
                                description_lines.append(line)
                            if len(description_lines) >= 3:
                                break
                        
                        if description_lines:
                            description = ' '.join(description_lines)
                            return description[:500] + '...' if len(description) > 500 else description
                except Exception as e:
                    logger.error(f"Error reading README: {e}")
        
        return "No description available"
    
    def _format_recent_commits(self, commits: List[Dict]) -> str:
        """Format recent commits for display"""
        if not commits:
            return "No recent commits available"
        
        formatted = []
        for commit in commits:
            formatted.append(f"- {commit['hash']}: {commit['message'][:100]}... ({commit['date']})")
        
        return "\n".join(formatted)
    
    def _format_file_structure(self, files: List[str]) -> str:
        """Format file structure for display"""
        return "\n".join([f"- {file}" for file in files])
    
    def _format_relevant_files(self, files: List[str]) -> str:
        """Format relevant files for display"""
        if not files:
            return "No specific files found for this query"
        
        return "\n".join([f"- {file}" for file in files])
    
    def chat(self, query: str) -> str:
        """Process chat query and return response"""
        try:
            # Get repository context
            context = self.get_repo_context(query)
            
            # Create the prompt
            prompt = f"""You are a helpful assistant that analyzes GitHub repositories. 
You have access to the following repository information:

{context}

Based on this repository context, please answer the following question:
{query}

Guidelines:
- Provide a comprehensive, accurate answer based on the repository information
- If you reference specific files or code, mention them clearly
- Be helpful and provide actionable insights when possible
- If the query is about setup or installation, look for requirements files, package.json, etc.
- If asked about the project structure, explain the main directories and their purposes
- If asked about functionality, explain what the code does based on the file contents
"""
            
            # Get response from OpenAI
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful GitHub repository analysis assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return f"An error occurred while processing your request: {str(e)}"

# Streamlit Web Interface
def main():
    st.set_page_config(
        page_title="GitHub Repository Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 GitHub Repository Chatbot")
    st.markdown("Chat with any GitHub repository! Supports both URLs and local paths.")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # OpenAI API Key
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            help="Enter your OpenAI API key"
        )
        
        # Repository Input
        repo_input = st.text_input(
            "Repository URL or Local Path",
            value="https://github.com/smantri-iitr/agentic_ai.git",
            help="Enter a GitHub URL or local repository path"
        )
        
        # Show input type
        if repo_input:
            if repo_input.startswith("http") or repo_input.startswith("git@"):
                st.info("📡 GitHub URL detected - will clone repository")
            else:
                st.info("📁 Local path detected")
        
        # Initialize chatbot button
        if st.button("Initialize Chatbot"):
            if api_key and repo_input:
                try:
                    with st.spinner("Initializing chatbot (this may take a moment for GitHub URLs)..."):
                        # Clean up previous chatbot if exists
                        if 'chatbot' in st.session_state:
                            st.session_state.chatbot.cleanup()
                        
                        st.session_state.chatbot = EnhancedGitHubChatbot(repo_input, api_key)
                        st.success("Chatbot initialized successfully!")
                        
                        # Show repository info
                        repo_info = st.session_state.chatbot._get_basic_repo_info()
                        st.info(f"Repository: {repo_info['name']}")
                        st.info(f"Files: {len(repo_info['files'])}")
                        st.info(f"Main Language: {repo_info['main_language']}")
                        if repo_info.get('remote_url'):
                            st.info(f"URL: {repo_info['remote_url']}")
                        
                except Exception as e:
                    st.error(f"Error initializing chatbot: {e}")
            else:
                st.error("Please provide both API key and repository input")
    
    # Main chat interface
    if 'chatbot' in st.session_state:
        st.header("Chat with your Repository")
        
        # Initialize chat history
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask about your repository..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get bot response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing repository..."):
                    response = st.session_state.chatbot.chat(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
    
    else:
        st.info("Please configure and initialize the chatbot using the sidebar.")
        
        # Show example queries
        st.header("Example Queries")
        example_queries = [
            "What is this repository about?",
            "How do I set up this project?",
            "What are the main components?",
            "Show me the project structure",
            "What dependencies are required?",
            "How do I run this project?",
            "What are the recent changes?",
            "Explain the main algorithms used"
        ]
        
        cols = st.columns(2)
        for i, query in enumerate(example_queries):
            with cols[i % 2]:
                st.code(query, language=None)

# CLI Interface
def cli_interface():
    """Command line interface for the chatbot"""
    print("Enhanced GitHub Repository Chatbot CLI")
    print("=" * 50)
    
    # Get configuration
    api_key = os.getenv("OPENAI_API_KEY")
    repo_input = os.getenv("REPO_INPUT")
    
    if not api_key:
        api_key = input("Enter OpenAI API Key: ")
    if not repo_input:
        repo_input = input("Enter GitHub URL or repository path: ")
    
    try:
        print("Initializing chatbot...")
        chatbot = EnhancedGitHubChatbot(repo_input, api_key)
        print(f"Chatbot initialized for repository: {repo_input}")
        print("Type 'quit' to exit\n")
        
        while True:
            query = input("You: ")
            if query.lower() in ['quit', 'exit']:
                break
            
            print("Bot: ", end="")
            response = chatbot.chat(query)
            print(response)
            print()
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'chatbot' in locals():
            chatbot.cleanup()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        cli_interface()
    else:
        main()
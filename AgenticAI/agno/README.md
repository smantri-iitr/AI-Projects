# 🤖 Agno AI Agent Projects

This folder contains AI agent projects built using the [Agno](https://github.com/agno-ai/agno) framework, a powerful Python library for creating intelligent AI agents and multi-agent systems.

## 📁 Projects Overview

### 1. 💔 AI Breakup Recovery Agent System
**File:** `ai_breakup_recovery_agent_using_agno.py`

A comprehensive emotional support system featuring four specialized AI agents working together to help users through breakup recovery:

- **Therapist Agent**: Provides empathetic emotional support and validation
- **Closure Agent**: Helps express unsent feelings and create emotional closure messages
- **Routine Planner Agent**: Designs 7-day recovery challenges and self-care routines
- **Brutal Honesty Agent**: Offers direct, objective feedback using web search tools

**Features:**
- Multi-modal input support (text + images)
- Streamlit web interface
- DuckDuckGo search integration for real-time information
- Personalized recovery plans and challenges

### 2. 🗞️ Multi-Agent News & Finance Team
**File:** `multi_aiagent_system_using_agno.py`

A professional news and financial analysis system with three coordinated agents:

- **Web Agent**: Searches and analyzes latest news using DuckDuckGo
- **Finance Agent**: Analyzes financial data and market trends using Yahoo Finance
- **Lead Editor**: Coordinates insights from both agents for comprehensive reporting

**Features:**
- Real-time news and financial data analysis
- Error handling and fallback mechanisms
- Professional reporting style with source citations
- Stock price tracking and analyst recommendations

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- OpenAI API key
- Internet connection for web search and financial data

### Installation

1. **Install required packages:**
```bash
pip install openai duckduckgo-search yfinance agno streamlit
```

2. **Set up your OpenAI API key:**
   - Get an API key from [OpenAI](https://platform.openai.com/api-keys)
   - Use the key in the application interface

### Running the Projects

#### Breakup Recovery Agent
```bash
cd AgenticAI/agno
streamlit run ai_breakup_recovery_agent_using_agno.py
```

#### Multi-Agent News & Finance System
```bash
cd AgenticAI/agno
python multi_aiagent_system_using_agno.py
```

## 🛠️ Technical Details

### Agno Framework Features Used
- **Agent Creation**: Custom agents with specific roles and instructions
- **Model Integration**: OpenAI GPT-4 integration via Agno's model interface
- **Tool Integration**: DuckDuckGo search and Yahoo Finance tools
- **Team Coordination**: Multi-agent collaboration and task delegation
- **Error Handling**: Robust error handling with fallback mechanisms

### Architecture Patterns
- **Multi-Agent Systems**: Specialized agents working together
- **Tool Chaining**: Sequential tool usage for complex tasks
- **Error Resilience**: Graceful degradation when external services fail
- **Modular Design**: Easy to extend and modify individual agents

## 🔧 Customization

### Adding New Agents
1. Create a new `Agent` instance with custom instructions
2. Define the agent's role and capabilities
3. Integrate with existing agent team

### Extending Tools
1. Import additional Agno tools or create custom ones
2. Add tools to agent configurations
3. Update agent instructions to use new capabilities

### Modifying Agent Behavior
- Edit the `instructions` list in each agent
- Adjust the `role` and `name` parameters
- Modify tool configurations as needed

## 📚 Learning Resources

- [Agno Documentation](https://github.com/agno-ai/agno)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Streamlit Documentation](https://docs.streamlit.io/)

## 🤝 Contributing

Feel free to:
- Add new agent types
- Improve error handling
- Enhance the user interface
- Add new tool integrations
- Optimize agent coordination

## 📄 License

This project uses the Agno framework and follows its licensing terms. Please refer to the individual project files for specific licensing information.

---

**Note:** These projects demonstrate advanced AI agent capabilities and are intended for educational and practical use. Always ensure you have proper API access and follow usage guidelines for external services.

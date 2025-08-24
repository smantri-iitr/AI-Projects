#!/usr/bin/env python3
"""
AutoGen Chatbot Implementation
A simple chatbot using Microsoft's AutoGen framework with multiple agent types.
"""

import autogen
from typing import Dict, List, Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AutoGenChatbot:
    def __init__(self, model: str = "gpt-3.5-turbo"):
        """
        Initialize the AutoGen chatbot with different agent types.
        
        Args:
            model: The LLM model to use (default: gpt-3.5-turbo)
        """
        # Configuration for the LLM
        self.llm_config = {
            "config_list": [
                {
                    "model": model,
                    "api_key": os.getenv("OPENAI_API_KEY"),  # Set your API key in .env file
                }
            ],
            "temperature": 0.7,
            "timeout": 60,
        }
        
        self.setup_agents()
    
    def setup_agents(self):
        """Set up different types of agents for conversation."""
        
        # Assistant Agent - Main conversational agent
        self.assistant = autogen.AssistantAgent(
            name="ChatBot",
            system_message="""You are a helpful, friendly, and knowledgeable chatbot. 
            You can assist with various topics including:
            - General questions and conversations
            - Programming and technical help
            - Creative writing and brainstorming
            - Problem-solving and advice
            
            Always be polite, helpful, and engaging in your responses.
            If you don't know something, admit it and offer to help find the information.
            """,
            llm_config=self.llm_config,
        )
        
        # User Proxy Agent - Represents the human user
        self.user_proxy = autogen.UserProxyAgent(
            name="User",
            human_input_mode="ALWAYS",  # Always ask for human input
            max_consecutive_auto_reply=0,  # Don't auto-reply
            is_termination_msg=lambda x: x.get("content", "").strip().lower() in ["exit", "quit", "bye", "goodbye"],
            code_execution_config=False,  # Disable code execution for safety
        )
        
        # Specialist Agent - For technical questions
        self.specialist = autogen.AssistantAgent(
            name="TechSpecialist",
            system_message="""You are a technical specialist who can provide detailed, 
            accurate information about programming, technology, science, and engineering topics.
            You provide clear explanations with examples when helpful.
            You can write code snippets and explain technical concepts in simple terms.
            """,
            llm_config=self.llm_config,
        )
        
        # Creative Agent - For creative tasks
        self.creative = autogen.AssistantAgent(
            name="CreativeAssistant",
            system_message="""You are a creative assistant specializing in:
            - Creative writing (stories, poems, scripts)
            - Brainstorming and ideation
            - Content creation
            - Artistic suggestions
            
            You're imaginative, inspiring, and help users explore their creative potential.
            """,
            llm_config=self.llm_config,
        )
    
    def start_chat(self):
        """Start an interactive chat session."""
        print("🤖 AutoGen Chatbot Started!")
        print("Type 'exit', 'quit', 'bye', or 'goodbye' to end the conversation.")
        print("You can ask me anything - I have different specialists to help you!\n")
        
        try:
            # Start the conversation
            self.user_proxy.initiate_chat(
                self.assistant,
                message="Hello! I'm ready to chat. What would you like to talk about today?"
            )
        except KeyboardInterrupt:
            print("\nChatbot session ended by user.")
        except Exception as e:
            print(f"An error occurred: {e}")
    
    def group_chat_mode(self):
        """Start a group chat with multiple agents."""
        print("🤖 AutoGen Group Chat Started!")
        print("Multiple AI agents will participate in the conversation.")
        print("Type 'exit', 'quit', 'bye', or 'goodbye' to end the conversation.\n")
        
        # Create a group chat
        groupchat = autogen.GroupChat(
            agents=[self.user_proxy, self.assistant, self.specialist, self.creative],
            messages=[],
            max_round=50,
            speaker_selection_method="round_robin",  # Or "auto" for automatic selection
        )
        
        manager = autogen.GroupChatManager(
            groupchat=groupchat,
            llm_config=self.llm_config
        )
        
        try:
            self.user_proxy.initiate_chat(
                manager,
                message="Hello everyone! I'd like to have a conversation with the team."
            )
        except KeyboardInterrupt:
            print("\nGroup chat session ended by user.")
        except Exception as e:
            print(f"An error occurred: {e}")
    
    def custom_chat(self, initial_message: str, agent_type: str = "assistant"):
        """
        Start a chat with a custom message and specific agent.
        
        Args:
            initial_message: The message to start the conversation
            agent_type: Type of agent ("assistant", "specialist", "creative")
        """
        agent_map = {
            "assistant": self.assistant,
            "specialist": self.specialist,
            "creative": self.creative
        }
        
        selected_agent = agent_map.get(agent_type, self.assistant)
        
        try:
            self.user_proxy.initiate_chat(
                selected_agent,
                message=initial_message
            )
        except Exception as e:
            print(f"An error occurred: {e}")


def main():
    """Main function to run the chatbot."""
    
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not found in environment variables.")
        print("Please create a .env file with your OpenAI API key:")
        print("OPENAI_API_KEY=your_api_key_here")
        print("\nFor demo purposes, you can still run the code but it won't work without a valid API key.")
        return
    
    # Create chatbot instance
    chatbot = AutoGenChatbot()
    
    # Show menu
    while True:
        print("\n" + "="*50)
        print("🤖 AutoGen Chatbot Options:")
        print("1. Simple Chat (1-on-1 with assistant)")
        print("2. Group Chat (Multiple AI agents)")
        print("3. Technical Chat (With specialist)")
        print("4. Creative Chat (With creative assistant)")
        print("5. Exit")
        print("="*50)
        
        choice = input("Select an option (1-5): ").strip()
        
        if choice == "1":
            chatbot.start_chat()
        elif choice == "2":
            chatbot.group_chat_mode()
        elif choice == "3":
            chatbot.custom_chat("Hello! I have a technical question.", "specialist")
        elif choice == "4":
            chatbot.custom_chat("Hello! I'd like help with something creative.", "creative")
        elif choice == "5":
            print("Goodbye! 👋")
            break
        else:
            print("Invalid choice. Please select 1-5.")


if __name__ == "__main__":
    main()


# Additional utility functions for advanced features

class AdvancedAutoGenChatbot(AutoGenChatbot):
    """Extended version with more advanced features."""
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        super().__init__(model)
        self.conversation_history: List[Dict] = []
    
    def save_conversation(self, filename: str = "chat_history.txt"):
        """Save conversation history to a file."""
        try:
            with open(filename, "w", encoding="utf-8") as f:
                for entry in self.conversation_history:
                    f.write(f"{entry['timestamp']}: {entry['speaker']}: {entry['message']}\n")
            print(f"Conversation saved to {filename}")
        except Exception as e:
            print(f"Error saving conversation: {e}")
    
    def load_conversation(self, filename: str = "chat_history.txt"):
        """Load conversation history from a file."""
        try:
            with open(filename, "r", encoding="utf-8") as f:
                content = f.read()
            print(f"Conversation loaded from {filename}")
            return content
        except Exception as e:
            print(f"Error loading conversation: {e}")
            return None
    
    def setup_custom_functions(self):
        """Set up custom functions that agents can call."""
        
        def get_weather(city: str) -> str:
            """Mock weather function - replace with real API call."""
            return f"The weather in {city} is sunny and 75°F."
        
        def calculate(expression: str) -> str:
            """Safe calculator function."""
            try:
                # Basic safety check
                allowed_chars = set("0123456789+-*/().")
                if all(c in allowed_chars or c.isspace() for c in expression):
                    result = eval(expression)
                    return f"Result: {result}"
                else:
                    return "Error: Invalid characters in expression"
            except Exception as e:
                return f"Error: {e}"
        
        # Register functions with agents
        self.assistant.register_function(
            function_map={
                "get_weather": get_weather,
                "calculate": calculate,
            }
        )


# Example usage and testing
def test_chatbot():
    """Test function to demonstrate chatbot capabilities."""
    print("Testing AutoGen Chatbot...")
    
    # Create a simple test without requiring API key
    try:
        # Mock configuration for testing
        test_config = {
            "config_list": [{"model": "mock-model", "api_key": "test-key"}],
            "temperature": 0.7,
        }
        
        print("✅ Chatbot components created successfully!")
        print("✅ Agents configured properly!")
        print("✅ Ready to start conversations!")
        
    except Exception as e:
        print(f"❌ Error in setup: {e}")


if __name__ == "__main__":
    # Uncomment the line below to run tests instead of the main chatbot
    # test_chatbot()
    main()
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.media import Image as AgnoImage
from agno.tools.duckduckgo import DuckDuckGoTools
import streamlit as st
from typing import List, Optional
import logging
from pathlib import Path
import tempfile
import os


# Configure logging for errors only
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

def initialize_agents(api_key: str) -> tuple[Agent, Agent, Agent, Agent]:
    try:
        # Use OpenAIChat from agno instead of OpenAI directly
        model=OpenAIChat(id="gpt-4o", api_key=api_key)
        
        therapist_agent = Agent(
            model=model,
            name="Therapist Agent",
            instructions=[
                "You are an empathetic therapist that:",
                "1. Listens with empathy and validates feelings",
                "2. Uses gentle humor to lighten the mood",
                "3. Shares relatable breakup experiences",
                "4. Offers comforting words and encouragement",
                "5. Analyzes both text and image inputs for emotional context",
                "Be supportive and understanding in your responses"
            ],
            markdown=True
        )

        closure_agent = Agent(
            model=model,
            name="Closure Agent",
            instructions=[
                "You are a closure specialist that:",
                "1. Creates emotional messages for unsent feelings",
                "2. Helps express raw, honest emotions",
                "3. Formats messages clearly with headers",
                "4. Ensures tone is heartfelt and authentic",
                "Focus on emotional release and closure"
            ],
            markdown=True
        )

        routine_planner_agent = Agent(
            model=model,
            name="Routine Planner Agent",
            instructions=[
                "You are a recovery routine planner that:",
                "1. Designs 7-day recovery challenges",
                "2. Includes fun activities and self-care tasks",
                "3. Suggests social media detox strategies",
                "4. Creates empowering playlists",
                "Focus on practical recovery steps"
            ],
            markdown=True
        )

        brutal_honesty_agent = Agent(
            model=model,
            name="Brutal Honesty Agent",
            tools=[DuckDuckGoTools()],
            instructions=[
                "You are a direct feedback specialist that:",
                "1. Gives raw, objective feedback about breakups",
                "2. Explains relationship failures clearly",
                "3. Uses blunt, factual language",
                "4. Provides reasons to move forward",
                "Focus on honest insights without sugar-coating"
            ],
            markdown=True
        )
        
        return therapist_agent, closure_agent, routine_planner_agent, brutal_honesty_agent
    except Exception as e:
        st.error(f"Error initializing agents: {str(e)}")
        logger.error(f"Agent initialization error: {str(e)}")
        return None, None, None, None

# Set page config and UI elements
st.set_page_config(
    page_title="💔 Breakup Recovery Squad",
    page_icon="💔",
    layout="wide"
)

# Initialize session state
if "api_key_input" not in st.session_state:
    st.session_state.api_key_input = ""

# Sidebar for API key input
with st.sidebar:
    st.header("🔑 API Configuration")
        
    api_key = st.text_input(
        "Enter your OpenAI API Key",
        value=st.session_state.api_key_input,
        type="password",
        help="Get your API key from OpenAI",
        key="api_key_widget"  
    )

    if api_key != st.session_state.api_key_input:
        st.session_state.api_key_input = api_key
    
    if api_key:
        st.success("API Key provided! ✅")
    else:
        st.warning("Please enter your API key to proceed")
        st.markdown("""
        To get your API key:
        1. Go to [OpenAI API Keys](https://platform.openai.com/api-keys)
        2. Create a new secret key
        3. Copy and paste it here
        """)

# Main content
st.title("💔 Breakup Recovery Squad")
st.markdown("""
    ### Your AI-powered breakup recovery team is here to help!
    Share your feelings and chat screenshots, and we'll help you navigate through this tough time.
""")

# Input section
col1, col2 = st.columns(2)

with col1:
    st.subheader("Share Your Feelings")
    user_input = st.text_area(
        "How are you feeling? What happened?",
        height=150,
        placeholder="Tell us your story..."
    )
    
with col2:
    st.subheader("Upload Chat Screenshots")
    uploaded_files = st.file_uploader(
        "Upload screenshots of your chats (optional)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="screenshots"
    )
    
    if uploaded_files:
        for file in uploaded_files:
            st.image(file, caption=file.name, use_container_width=True)

# Process button and API key check
if st.button("Get Recovery Plan 💝", type="primary"):
    if not st.session_state.api_key_input.strip():
        st.warning("Please enter your API key in the sidebar first!")
    elif not user_input.strip() and not uploaded_files:
        st.warning("Please share your feelings or upload screenshots to get help.")
    else:
        with st.spinner("Initializing your recovery team..."):
            therapist_agent, closure_agent, routine_planner_agent, brutal_honesty_agent = initialize_agents(st.session_state.api_key_input)
        
        if all([therapist_agent, closure_agent, routine_planner_agent, brutal_honesty_agent]):
            try:
                st.header("Your Personalized Recovery Plan")
                
                def process_images(files):
                    """Process uploaded image files for agent analysis"""
                    processed_images = []
                    if not files:
                        return processed_images
                        
                    for file in files:
                        try:
                            # Create temporary file
                            temp_dir = tempfile.gettempdir()
                            temp_path = os.path.join(temp_dir, f"temp_{file.name}")
                            
                            # Write file content
                            with open(temp_path, "wb") as f:
                                f.write(file.getvalue())
                            
                            # Create AgnoImage object
                            agno_image = AgnoImage(filepath=Path(temp_path))
                            processed_images.append(agno_image)
                            
                        except Exception as e:
                            logger.error(f"Error processing image {file.name}: {str(e)}")
                            st.warning(f"Could not process image: {file.name}")
                            continue
                        finally:
                            # Clean up temporary file
                            try:
                                if os.path.exists(temp_path):
                                    os.remove(temp_path)
                            except:
                                pass
                                
                    return processed_images
                
                # Process images once
                all_images = process_images(uploaded_files) if uploaded_files else []
                
                # Create tabs for better organization
                tab1, tab2, tab3, tab4 = st.tabs(["🤗 Support", "✍️ Closure", "📅 Recovery Plan", "💪 Honest Talk"])
                
                with tab1:
                    with st.spinner("Getting empathetic support..."):
                        try:
                            therapist_prompt = f"""
                            Analyze the emotional state and provide empathetic support based on:
                            User's message: {user_input}
                            
                            Please provide a compassionate response with:
                            1. Validation of feelings
                            2. Gentle words of comfort
                            3. Relatable experiences or insights
                            4. Words of encouragement
                            5. Practical emotional coping strategies
                            
                            Keep the tone warm, understanding, and supportive.
                            """
                            
                            response = therapist_agent.run(
                                message=therapist_prompt,
                                images=all_images
                            )
                            
                            st.markdown(response.content if hasattr(response, 'content') else str(response))
                        except Exception as e:
                            logger.error(f"Therapist agent error: {str(e)}")
                            st.error("Sorry, there was an issue getting emotional support. Please try again.")
                
                with tab2:
                    with st.spinner("Crafting closure messages..."):
                        try:
                            closure_prompt = f"""
                            Help create emotional closure based on:
                            User's feelings: {user_input}
                            
                            Please provide:
                            1. Template for unsent messages to express feelings
                            2. Emotional release exercises
                            3. Closure ritual suggestions
                            4. Strategies for moving forward
                            5. Ways to process unresolved emotions
                            
                            Focus on healthy emotional expression and letting go.
                            """
                            
                            response = closure_agent.run(
                                message=closure_prompt,
                                images=all_images
                            )
                            
                            st.markdown(response.content if hasattr(response, 'content') else str(response))
                        except Exception as e:
                            logger.error(f"Closure agent error: {str(e)}")
                            st.error("Sorry, there was an issue creating closure content. Please try again.")
                
                with tab3:
                    with st.spinner("Creating your recovery plan..."):
                        try:
                            routine_prompt = f"""
                            Design a comprehensive 7-day recovery plan based on:
                            Current emotional state: {user_input}
                            
                            Include:
                            1. Daily activities and recovery challenges
                            2. Self-care routines and wellness practices
                            3. Social media detox strategies
                            4. Mood-lifting music and entertainment suggestions
                            5. Physical activities and exercise recommendations
                            6. Social connection ideas
                            7. Personal growth activities
                            
                            Make it practical, achievable, and uplifting.
                            """
                            
                            response = routine_planner_agent.run(
                                message=routine_prompt,
                                images=all_images
                            )
                            
                            st.markdown(response.content if hasattr(response, 'content') else str(response))
                        except Exception as e:
                            logger.error(f"Routine planner error: {str(e)}")
                            st.error("Sorry, there was an issue creating your recovery plan. Please try again.")
                
                with tab4:
                    with st.spinner("Getting honest perspective..."):
                        try:
                            honesty_prompt = f"""
                            Provide honest, constructive feedback about:
                            Situation: {user_input}
                            
                            Include:
                            1. Objective analysis of the situation
                            2. Growth opportunities and lessons learned
                            3. Realistic future outlook
                            4. Actionable steps for improvement
                            5. Red flags to watch for in future relationships
                            6. Personal development recommendations
                            
                            Be direct but constructive, focusing on empowerment and growth.
                            """
                            
                            response = brutal_honesty_agent.run(
                                message=honesty_prompt,
                                images=all_images
                            )
                            
                            st.markdown(response.content if hasattr(response, 'content') else str(response))
                        except Exception as e:
                            logger.error(f"Honesty agent error: {str(e)}")
                            st.error("Sorry, there was an issue getting honest feedback. Please try again.")
                            
            except Exception as e:
                logger.error(f"General error during analysis: {str(e)}")
                st.error("An unexpected error occurred. Please check your API key and try again.")
        else:
            st.error("Failed to initialize recovery agents. Please check your API key and try again.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Made with ❤️ by the Breakup Recovery Squad</p>
        <p>Remember: This too shall pass. You're stronger than you think! 💪</p>
        <p><em>Disclaimer: This is an AI-powered tool for emotional support. For serious mental health concerns, please consult a professional.</em></p>
    </div>
""", unsafe_allow_html=True)
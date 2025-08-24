"""🗞️ Multi-Agent Team - Your Professional News & Finance Squad!

This example shows how to create a powerful team of AI agents working together
to provide comprehensive financial analysis and news reporting. The team consists of:
1. Web Agent: Searches and analyzes latest news
2. Finance Agent: Analyzes financial data and market trends
3. Lead Editor: Coordinates and combines insights from both agents

Run: `pip install openai duckduckgo-search yfinance agno` to install the dependencies
"""

import logging
from textwrap import dedent
import sys

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.team.team import Team
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools

# Configure logging to help debug issues
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom DuckDuckGo wrapper with error handling
class SafeDuckDuckGoTools(DuckDuckGoTools):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def search_news(self, query: str, max_results: int = 5):
        """Search news with error handling"""
        try:
            return super().search_news(query, max_results)
        except Exception as e:
            logger.warning(f"DuckDuckGo news search failed: {e}")
            return f"News search temporarily unavailable for '{query}'. Please try a more specific search term or try again later."
    
    def search(self, query: str, max_results: int = 5):
        """Search web with error handling"""
        try:
            return super().search(query, max_results)
        except Exception as e:
            logger.warning(f"DuckDuckGo web search failed: {e}")
            return f"Web search temporarily unavailable for '{query}'. Please try a more specific search term or try again later."

# Custom YFinance wrapper with error handling
class SafeYFinanceTools(YFinanceTools):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def get_stock_price(self, symbol: str):
        """Get stock price with error handling"""
        try:
            return super().get_stock_price(symbol)
        except Exception as e:
            logger.warning(f"YFinance stock price failed for {symbol}: {e}")
            return f"Stock price data temporarily unavailable for {symbol}. Please verify the ticker symbol and try again."
    
    def get_analyst_recommendations(self, symbol: str):
        """Get analyst recommendations with error handling"""
        try:
            return super().get_analyst_recommendations(symbol)
        except Exception as e:
            logger.warning(f"YFinance analyst recommendations failed for {symbol}: {e}")
            return f"Analyst recommendations temporarily unavailable for {symbol}. Please verify the ticker symbol and try again."
    
    def get_company_info(self, symbol: str):
        """Get company info with error handling"""
        try:
            return super().get_company_info(symbol)
        except Exception as e:
            logger.warning(f"YFinance company info failed for {symbol}: {e}")
            return f"Company information temporarily unavailable for {symbol}. Please verify the ticker symbol and try again."

web_agent = Agent(
    name="Web Agent",
    role="Search the web for information",
    model=OpenAIChat(id="gpt-4o"),
    tools=[SafeDuckDuckGoTools()],
    instructions=dedent("""\
        You are an experienced web researcher and news analyst! 🔍

        Follow these steps when searching for information:
        1. Start with the most recent and relevant sources
        2. Cross-reference information from multiple sources when possible
        3. Prioritize reputable news outlets and official sources
        4. Always cite your sources with links when available
        5. Focus on market-moving news and significant developments

        Error Handling:
        - If search tools are temporarily unavailable, acknowledge this and provide general market context
        - Suggest alternative approaches when specific searches fail
        - Always inform users about data limitations

        Your style guide:
        - Present information in a clear, journalistic style
        - Use bullet points for key takeaways
        - Include relevant quotes when available
        - Specify the date and time for each piece of news when possible
        - Highlight market sentiment and industry trends
        - End with a brief analysis of the overall narrative
        - Pay special attention to regulatory news, earnings reports, and strategic announcements\
    """),
    show_tool_calls=True,
    markdown=True,
)

finance_agent = Agent(
    name="Finance Agent",
    role="Get financial data",
    model=OpenAIChat(id="gpt-4o"),
    tools=[
        SafeYFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True)
    ],
    instructions=dedent("""\
        You are a skilled financial analyst with expertise in market data! 📊

        Follow these steps when analyzing financial data:
        1. Start with the latest stock price, trading volume, and daily range
        2. Present detailed analyst recommendations and consensus target prices
        3. Include key metrics: P/E ratio, market cap, 52-week range when available
        4. Analyze trading patterns and volume trends
        5. Compare performance against relevant sector indices when possible

        Error Handling:
        - If financial data is temporarily unavailable, acknowledge this clearly
        - Provide historical context or general market information as fallback
        - Suggest retrying with verified ticker symbols if data retrieval fails

        Your style guide:
        - Use tables for structured data presentation when data is available
        - Include clear headers for each data section
        - Add brief explanations for technical terms
        - Highlight notable changes with emojis (📈 📉)
        - Use bullet points for quick insights
        - Compare current values with historical averages when possible
        - End with a data-driven financial outlook or acknowledge data limitations\
    """),
    show_tool_calls=True,
    markdown=True,
)

agent_team = Team(
    members=[web_agent, finance_agent],
    model=OpenAIChat(id="gpt-4o"),
    mode="coordinate",
    success_criteria=dedent("""\
        A comprehensive financial news report with clear sections and available data-driven insights.
        If data is unavailable, provide clear acknowledgment and alternative analysis approaches.
    """),
    instructions=dedent("""\
        You are the lead editor of a prestigious financial news desk! 📰

        Your role:
        1. Coordinate between the web researcher and financial analyst
        2. Combine their findings into a compelling narrative
        3. Ensure all information is properly sourced and verified
        4. Present a balanced view of both news and data
        5. Highlight key risks and opportunities
        6. Handle data unavailability gracefully

        Quality Control:
        - Always acknowledge when data sources are temporarily unavailable
        - Provide alternative analysis methods when primary tools fail
        - Maintain professional reporting standards regardless of data limitations
        - Suggest follow-up actions when appropriate

        Your style guide:
        - Start with an attention-grabbing headline
        - Begin with a powerful executive summary
        - Present available financial data first, followed by news context
        - Use clear section breaks between different types of information
        - Include relevant charts or tables when data is available
        - Add 'Market Sentiment' section with current mood when possible
        - Include a 'Key Takeaways' section at the end
        - Add 'Data Limitations' section if tools were unavailable
        - End with 'Risk Factors' when appropriate
        - Sign off with 'Market Watch Team' and the current date\
    """),
    add_datetime_to_instructions=True,
    show_tool_calls=True,
    markdown=True,
    enable_agentic_context=True,
    show_members_responses=False,
)

def run_analysis(query: str):
    """Run analysis with error handling"""
    try:
        print(f"\n🚀 Starting analysis for: {query}")
        print("=" * 60)
        agent_team.print_response(
            message=query,
            stream=True,
        )
    except Exception as e:
        print(f"\n❌ Analysis failed with error: {e}")
        print("Please check your internet connection and API keys, then try again.")
        logger.error(f"Analysis failed: {e}", exc_info=True)

# Example usage with error handling
if __name__ == "__main__":
    # Test with the original query
    run_analysis("Summarize analyst recommendations and share the latest news for NVDA")
    
    # Additional example queries to try:
    example_queries = [
        "What's the market outlook and financial performance of AI semiconductor companies?",
        "Analyze recent developments and financial performance of TSLA",
        "Compare the financial performance and recent news of major cloud providers (AMZN, MSFT, GOOGL)",
        "What's the impact of recent Fed decisions on banking stocks? Focus on JPM and BAC",
        "Analyze the gaming industry outlook through ATVI, EA, and TTWO performance",
        "How are social media companies performing? Compare META and SNAP",
        "What's the latest on AI chip manufacturers and their market position?"
    ]
    
    # Uncomment to run additional queries
    # for query in example_queries:
    #     run_analysis(query)
    #     print("\n" + "="*80 + "\n")

print("\n📝 Additional Configuration Tips:")
print("1. Ensure you have stable internet connection")
print("2. Verify your OpenAI API key is properly set")
print("3. Try running with fewer concurrent requests if you encounter rate limits")
print("4. Consider adding delays between requests for better stability")
print("\n💡 If you continue to experience issues:")
print("- Check if DuckDuckGo search is accessible from your location")
print("- Verify YFinance can access market data")
print("- Consider using alternative search tools if available")
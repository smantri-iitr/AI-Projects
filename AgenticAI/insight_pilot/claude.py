import os
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_anthropic import ChatAnthropic
from langchain_community.agent_toolkits import create_sql_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def create_agent(query):

    db_path = os.path.expanduser('/Users/shubhammantri/Downloads/random/freelancing_work/sample_data.duckdb')
    engine = create_engine(f'duckdb:///{db_path}')
    db = SQLDatabase(engine)

    llm = ChatAnthropic(model='claude-sonnet-4-20250514', temperature=0, timeout=300, stop=None)
    with open('./prompt.txt') as fp:
        claude_prompt = fp.read()
        prompt = ChatPromptTemplate.from_messages(
            [('system', claude_prompt), ('human', '{input}'), MessagesPlaceholder(variable_name='agent_scratchpad')]
        )

    agent = create_sql_agent(
        llm=llm,
        db=db,
        agent_type='tool-calling',
        verbose=True,
        prompt=prompt,
    )
    return agent.run(query)[0]['text']


if __name__ == '__main__':
    # if you run this file directly then put the query for the agent down below
    query = 'ENTER YOUR QUERY HERE'
    print(create_agent(query))
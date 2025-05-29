import operator
import os
from typing import Annotated, List, Literal, Sequence, TypedDict

from langchain.pydantic_v1 import (  # Using Pydantic v1 for Langchain tool compatibility
    BaseModel,
    Field,
)
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent

from dbt.skills.curriculum_data import create_dbt_framework

# Assuming dbt.skills modules are in the PYTHONPATH or same directory structure
# If running this script directly from a directory that doesn't have 'dbt' as a subdir,
# you might need to adjust python path. For example:
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # If dbt_supervisor_main.py is in a 'scripts' dir and 'dbt' is sibling to 'scripts'
from dbt.skills.schemas import (  # DBTSkill, SubSkill are used indirectly
    DBTFramework,
    DBTModule,
)

# --- Environment Setup (User needs to set this) ---
# Make sure to set your OPENAI_API_KEY in your environment variables
# e.g., export OPENAI_API_KEY="sk-..."
# For LangSmith tracing (optional, but recommended):
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = "ls_..."
# os.environ["LANGCHAIN_PROJECT"] = "DBT Multi-Agent"


# --- 1. Load DBT Framework ---
dbt_framework: DBTFramework = create_dbt_framework()
dbt_modules_dict: dict[str, DBTModule] = {
    module.name: module for module in dbt_framework.modules
}

# --- LLM Definition ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)
# For higher quality, consider "gpt-4-turbo"


# --- Helper function to format skills for prompts ---
def format_skills_for_prompt(module: DBTModule) -> str:
    skill_texts = []
    for skill in module.skills:
        text = f"- Skill: {skill.name} (Category: {skill.category})\n  Description: {skill.description}"
        if skill.sub_skills:
            text += "\n  Sub-skills:\n"
            for sub_skill in skill.sub_skills:
                text += f"    • {sub_skill.name}: {sub_skill.description}\n"
        skill_texts.append(text)
    return "\n".join(skill_texts)


# --- 2. Define Worker Agents ---
worker_agents_runnables: dict[str, any] = {}
member_names: List[str] = []

for module in dbt_framework.modules:
    module_name_str = module.name
    agent_name = f"{module_name_str.lower().replace(' ', '_').replace('-', '_')}_agent"
    member_names.append(agent_name)

    skill_details = format_skills_for_prompt(module)

    worker_system_prompt_string = (
        f"You are a specialized DBT agent expert in the '{module.name}' module.\n"
        "Your primary goal is to analyze the user's scenario (provided as the latest human message) "
        f"and recommend the most relevant skill(s) and sub-skill(s) strictly from the '{module.name}' module.\n"
        "Provide a clear and concise explanation for your recommendations.\n\n"
        f"Available skills in the '{module.name}' module:\n{skill_details}\n\n"
        "INSTRUCTIONS:\n"
        f"- ONLY recommend skills from the '{module.name}' module. Do not refer to skills from other modules.\n"
        f"- If the user's scenario does not seem relevant to the '{module.name}' module, clearly state that this module's skills may not be the best fit and briefly explain why.\n"
        "- Structure your response: First, state the recommended skill(s). Second, explain the relevance to the scenario. Example: 'Recommended Skill(s): Observe, Describe. Explanation: These skills can help by...'.\n"
        "- Respond ONLY with your analysis and recommendations. Do NOT include conversational fluff like 'Okay, I will analyze that' or 'Here is my analysis'.\n"
        "- Your response will be reviewed by a supervisor. Ensure it is complete and directly addresses the task."
    )

    agent_runnable = create_react_agent(
        llm,
        tools=[],  # No external tools for workers in this version
        prompt=worker_system_prompt_string,
        # name=agent_name, # create_react_agent does not take a name param for the agent itself, but for the AIMessage it produces if part of a chain
    )
    worker_agents_runnables[agent_name] = agent_runnable


# --- 3. Define State for the Graph ---
class SupervisorState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_worker: str  # The name of the next worker or "FINISH"
    user_scenario: str  # The initial user query for context


# --- 4. Create the Supervisor Agent ---
# Define the choices for the supervisor dynamically
supervisor_choices_list = member_names + ["FINISH"]
SupervisorChoicesLiteral = Literal[tuple(supervisor_choices_list)]  # type: ignore


# Pydantic model for the supervisor's structured output
class SupervisorRouter(BaseModel):
    next_worker: SupervisorChoicesLiteral = Field(  # type: ignore
        description=f"The next worker agent to delegate the task to, or 'FINISH' to end the process. Must be one of {supervisor_choices_list}."
    )


supervisor_system_prompt = (
    "You are a DBT Triage Supervisor. Your role is to manage a conversation and delegate tasks to specialized DBT worker agents "
    "or to conclude the interaction by summarizing the findings.\n"
    "The available worker agents are: {members_string}. Each agent is an expert in a specific DBT module.\n"
    "Based on the user's scenario (from the initial human message) and any subsequent analysis from worker agents (in AIMessages), "
    "determine the next step.\n\n"
    "DELEGATION RULES (when analyzing initial user query):\n"
    "- For present-moment awareness, observation, non-judgmental focus: delegate to the 'mindfulness_agent'.\n"
    "- For understanding, managing, or changing emotions effectively: delegate to the 'emotion_regulation_agent'.\n"
    "- For surviving crisis situations, accepting reality when unchangeable: delegate to the 'distress_tolerance_agent'.\n"
    "- For improving relationships, asking for needs, setting boundaries, saying no: delegate to the 'interpersonal_effectiveness_agent'.\n\n"
    "POST-WORKER ANALYSIS RULES:\n"
    "- If a worker agent has provided its analysis (visible as the last AIMessage from an agent), your task is to synthesize this information and provide a final, consolidated response to the user. In this case, choose 'FINISH'.\n"
    "- If the initial query is vague or a worker indicates the scenario is not relevant to its module, you should choose 'FINISH' and explain that no specific module seems directly applicable or more information is needed.\n\n"
    "OUTPUT FORMAT:\n"
    "Your response MUST be a JSON object with a single key 'next_worker', specifying the chosen agent's name or 'FINISH'.\n"
    'Example for delegation: {{ "next_worker": "mindfulness_agent" }}\n'
    'Example for concluding: {{ "next_worker": "FINISH" }}'
).format(members_string=", ".join(member_names))

supervisor_llm_with_structured_output = llm.with_structured_output(
    SupervisorRouter,
    method="json_mode",
)

supervisor_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", supervisor_system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
supervisor_runnable = supervisor_prompt_template | supervisor_llm_with_structured_output

# --- 5. Define Graph Nodes and Edges ---
graph_builder = StateGraph(SupervisorState)


# Supervisor Node Function
def supervisor_node_func(state: SupervisorState) -> dict:
    print("\n---SUPERVISOR ACTING---")
    supervisor_decision_obj: SupervisorRouter = supervisor_runnable.invoke(
        {"messages": state["messages"]}
    )
    next_worker_choice = supervisor_decision_obj.next_worker
    print(f"Supervisor decision: Delegate to '{next_worker_choice}'")

    if next_worker_choice == "FINISH":
        print("Supervisor is generating final summary...")
        worker_response_content = "No specific worker analysis was performed or found."
        if len(state["messages"]) > 1:
            last_message = state["messages"][-1]
            if (
                isinstance(last_message, AIMessage)
                and hasattr(last_message, "name")
                and last_message.name in member_names
            ):
                worker_response_content = f"The {last_message.name.replace('_agent', '')} agent provided the following analysis: '{last_message.content}'"

        final_summary_prompt_text = (
            "You are the main supervisor. The user described a scenario, and a relevant DBT worker agent (if applicable) has provided its analysis. "
            "Your task is to formulate a final, consolidated, and user-friendly DBT skill recommendation. "
            "Present the recommendation clearly. If a worker provided input, synthesize it. "
            "If no specific worker was engaged or if the worker found no relevant skills, acknowledge that and perhaps suggest general DBT principles if appropriate or state that no specific recommendation can be made based on the input.\n\n"
            f"User's initial scenario: '{state['user_scenario']}'\n"
            f"Worker's input (if any): {worker_response_content}\n\n"
            "Provide your final, helpful response to the user. Be direct and start with the recommendation or conclusion."
        )
        final_response_message: AIMessage = llm.invoke(
            [HumanMessage(content=final_summary_prompt_text)]
        )
        final_response_message.name = "supervisor_final_summary"
        print(f"Supervisor final summary: {final_response_message.content}")
        return {"messages": [final_response_message], "next_worker": "FINISH"}
    else:
        return {
            "next_worker": next_worker_choice,
            "messages": [],
        }  # No message from supervisor on delegation


graph_builder.add_node("supervisor", supervisor_node_func)


# Worker Node Function Factory
def create_worker_node_func(agent_name: str, agent_runnable_instance):
    def worker_node_func(state: SupervisorState) -> dict:
        print(f"\n---{agent_name.upper()} ACTING---")
        # The react agent from create_react_agent expects a list of messages
        # We only pass the user's scenario to keep it focused.
        task_messages = [
            HumanMessage(
                content=f"Please analyze this scenario: {state['user_scenario']}"
            )
        ]

        worker_response_dict = agent_runnable_instance.invoke(
            {"messages": task_messages}
        )

        ai_message_from_worker = AIMessage(
            content="Worker agent did not produce a structured response.",
            name=agent_name,
        )
        if worker_response_dict and "messages" in worker_response_dict:
            for m in reversed(worker_response_dict["messages"]):
                if isinstance(
                    m, AIMessage
                ):  # The react agent's final output is an AIMessage
                    ai_message_from_worker = AIMessage(
                        content=m.content, name=agent_name
                    )
                    break

        print(f"{agent_name} response: {ai_message_from_worker.content}")
        return {"messages": [ai_message_from_worker]}

    return worker_node_func


for name, runnable_instance in worker_agents_runnables.items():
    graph_builder.add_node(name, create_worker_node_func(name, runnable_instance))


# Conditional Edges from Supervisor
def route_from_supervisor(state: SupervisorState) -> str:
    next_node_chosen_by_supervisor = state.get("next_worker")
    if not next_node_chosen_by_supervisor:
        return END

    if next_node_chosen_by_supervisor == "FINISH":
        return END

    if next_node_chosen_by_supervisor in member_names:
        return next_node_chosen_by_supervisor
    else:
        return END


conditional_edge_mapping = {name: name for name in member_names}
conditional_edge_mapping[END] = END  # type: ignore

graph_builder.add_conditional_edges(
    "supervisor", route_from_supervisor, conditional_edge_mapping
)

# Edges from Workers back to Supervisor
for name in member_names:
    graph_builder.add_edge(name, "supervisor")

# Set Entry Point
graph_builder.set_entry_point("supervisor")

# Compile the graph
dbt_multi_agent_graph = graph_builder.compile()


# --- 6. Run the Graph ---
def run_dbt_assistant(user_query: str):
    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "Error: OPENAI_API_KEY environment variable is not set. Cannot run the assistant."
        )
        return

    print(f"\n🚀 Running DBT Multi-Agent Assistant for query: '{user_query}'")
    initial_state = {
        "messages": [HumanMessage(content=user_query)],
        "user_scenario": user_query,
        "next_worker": "",  # Initialize next_worker
    }

    final_graph_state = dbt_multi_agent_graph.invoke(
        initial_state, config={"recursion_limit": 15}
    )

    print("\n---FINAL GRAPH STATE (Simplified)---")
    if final_graph_state and final_graph_state.get("messages"):
        for msg in final_graph_state["messages"]:
            role = "User" if isinstance(msg, HumanMessage) else "AI"
            name_part = f" ({msg.name})" if hasattr(msg, "name") and msg.name else ""
            print(
                f"  {role}{name_part}: {msg.content[:100]}..."
            )  # Print first 100 chars

    print("\n✅ ---FINAL RESPONSE TO USER--- ✅")
    final_message_to_user = "Could not determine a final response."
    if final_graph_state and final_graph_state.get("messages"):
        last_msg_in_history = final_graph_state["messages"][-1]
        if (
            isinstance(last_msg_in_history, AIMessage)
            and hasattr(last_msg_in_history, "name")
            and last_msg_in_history.name == "supervisor_final_summary"
        ):
            final_message_to_user = last_msg_in_history.content
        elif isinstance(last_msg_in_history, AIMessage):
            final_message_to_user = f"The process concluded with the following from {getattr(last_msg_in_history, 'name', 'AI')}: {last_msg_in_history.content}"
        elif isinstance(last_msg_in_history, HumanMessage):
            final_message_to_user = "It seems the process ended on the initial user query without a supervisor summary."

    print(final_message_to_user)
    print("\n" + "=" * 50 + "\n")


# --- Example Usage ---
if __name__ == "__main__":
    print("Starting DBT Multi-Agent Assistant Demo")

    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "\nError: OPENAI_API_KEY is not set. Please set it as an environment variable. Exiting."
        )
    else:
        print("OPENAI_API_KEY found.")
        example_queries = [
            "I'm constantly feeling overwhelmed and find it hard to focus on one thing at a time. I'm always distracted by my thoughts.",
            "I got into a big argument with my friend because I felt they weren't listening to me, and I ended up yelling. Now I feel terrible and don't know how to fix it with them.",
            "I'm facing a really tough situation at work that I absolutely cannot change right now, and it's causing me immense stress. I just need to get through it without breaking down.",
            "I often feel like my emotions are a rollercoaster I can't control. One minute I'm fine, the next I'm incredibly sad or angry for no clear reason.",
            "I don't know what DBT is.",  # Example of something less specific
        ]

        run_dbt_assistant(example_queries[0])
        run_dbt_assistant(example_queries[1])
        run_dbt_assistant(example_queries[2])
        run_dbt_assistant(example_queries[3])
        run_dbt_assistant(example_queries[4])

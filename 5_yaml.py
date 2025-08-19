from crewai import Task, Agent, Process, LLM, Crew
from crewai.project import CrewBase, agent, crew, task
import os
import yaml

from crewai_tools import SerperDevTool, ScrapeWebsiteTool, DirectoryReadTool
from dotenv import load_dotenv

# ---------------------------
# Setup
# ---------------------------
load_dotenv(override=True)

print("Current Working Directory (PWD):", os.getcwd())
print("This file location (__file__):", __file__)
print("Absolute path of this script:", os.path.abspath(__file__))
print("Directory of this script:", os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AGENTS_PATH = os.path.join(BASE_DIR, "config", "agents.yaml")
TASKS_PATH = os.path.join(BASE_DIR, "config", "tasks.yaml")

# Load YAML configs as dicts
with open(AGENTS_PATH, "r") as f:
    agents_yaml = yaml.safe_load(f)

with open(TASKS_PATH, "r") as f:
    tasks_yaml = yaml.safe_load(f)

print("Agents YAML:", agents_yaml)
print("Tasks YAML:", tasks_yaml)


# ---------------------------
# Crew Definition
# ---------------------------
@CrewBase
class BlogCrew():
    """A crew for managing blog content creation and publication."""

    agents_config = AGENTS_PATH
    tasks_config = TASKS_PATH

    @agent
    def researcher_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["researcher_agent"],
            tools=[SerperDevTool()],
            verbose=True,
        )

    @agent
    def writer_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["writer_agent"],
            verbose=True,
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config["research_task"],
            agent=self.researcher_agent(),
            verbose=True,
        )

    @task
    def creative_task(self) -> Task:
        return Task(
            config=self.tasks_config["creative_task"],
            agent=self.writer_agent(),
            verbose=True,
        )

    @crew
    def blog_crew(self) -> Crew:
        return Crew(
            agents=[self.researcher_agent(), self.writer_agent()],
            tasks=[self.research_task(), self.creative_task()],
            verbose=True,
        )


# ---------------------------
# Run the Crew
# ---------------------------
if __name__ == "__main__":
    blog_crew = BlogCrew()
    result = blog_crew.blog_crew().kickoff(inputs={"topic": "The future of Electric Vehicles"})
    print("\n--- Final Result ---\n")
    print(result)

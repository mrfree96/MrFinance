from crewai import Agent, Task, Crew, Process

class QueryWorkFlow:
    
    def __init__(self, query_prompt):
        self.query_prompt = query_prompt
        self.agents = []
        self.tasks = []
        self.crew = []
        
    def create_agent(self, role, goal, backstory, llm, verbose=True, allow_delegation=False):
        agent = Agent(role=role, goal=goal, backstory=backstory, llm= llm,verbose=verbose, allow_delegation=allow_delegation)
        self.agents.append(agent)
        return agent
        
    def create_task(self, description, agent, expected_output):
        task = Task(description = description, agent=agent, expected_output=expected_output)
        self.tasks.append(task)
        return task
    
    def create_crew(self, verbose=True, process=Process.sequential, max_rpm=29):
        crewai_agents = [agent for agent in self.agents]
        crewai_tasks = [task for task in self.tasks]
        crew = Crew(
            agents = crewai_agents,
            tasks = crewai_tasks,
            verbose= verbose,
            process= process,
            max_rpm=max_rpm
        )
        return crew
    
    def change_prompt(self, new_prompt):
        self.query_prompt = new_prompt
        print(f"New prompt is: {self.query_prompt}")
        
    def get_agent():
        pass
    def get_task():
        pass
    def get_crew():
        pass
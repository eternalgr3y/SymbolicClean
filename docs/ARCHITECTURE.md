# System Architecture

This document provides a high-level overview of the SymbolicAGI's architecture, focusing on the flow of information and the responsibilities of each core component during a typical goal-execution cycle.

## Core Philosophy

The architecture is designed around a central **Orchestrator** (`SymbolicAGI`) that manages several specialized, subordinate functional units. This hub-and-spoke model promotes a clear separation of concerns:

-   **Executive Function (`SymbolicAGI`):** Holds state and dependencies. Manages the lifecycle of other units.
-   **Cognition & Planning (`Planner`, `RecursiveIntrospector`):** Handles abstract reasoning, goal decomposition, and self-reflection.
-   **Execution & Action (`ExecutionUnit`, `ToolPlugin`, `Agent`):** Performs concrete actions in the world or within the system.
-   **Memory & Identity (`SymbolicMemory`, `LongTermMemory`, `SymbolicIdentity`, `Consciousness`):** Provides the persistent state, memory, and value systems that ground the AGI's behavior.
-   **Governance (`EthicalEvaluator`):** Acts as a universal check on all planned actions and self-modifications.

## Data Flow: From Goal to Action

The following diagram illustrates the primary data flow when a new goal is introduced to the system. The loop is now enhanced with feedback from the AGI's internal state (emotions, trust scores).

```mermaid
graph TD
    A[User/System provides Goal] --> B(LongTermMemory);
    B --> C{ExecutionUnit};
    C --> D[Planner];

    subgraph "Cognitive State Feedback"
        M(EmotionalState) --> E;
        N(AgentPool Trust Scores) --> E;
    end

    E(RecursiveIntrospector) -- Injects State --> D;
    D --> E;

    D --> F{Plan};
    F --> G[EthicalEvaluator];
    G -- Plan Approved --> C;
    C -- Delegate Task (to most trusted agent) --> H(MessageBus);
    H -- Task Message --> I[Agent];
    I -- Perform Action --> J[Result];
    J -- Success/Failure --> C;
    C -- Updates Trust --> N;
    C -- Updates Emotions --> M;
    J -- Result Message --> H;
    H -- Result to Orchestrator --> C;
    C -- Update State --> K(SymbolicMemory / LTM);

Component Breakdown
Goal Ingestion: A new GoalModel is added to the LongTermMemory (LTM).
Cognitive Cycle: The ExecutionUnit's main loop detects the active goal.
State-Aware Planning: The ExecutionUnit invokes the Planner.
The RecursiveIntrospector gathers the AGI's current EmotionalState and cognitive_energy.
This internal state is injected into the system prompt for the Planner.
The Planner reasons about the goal, available tools, and its own internal state to produce a structured PlannerOutput.
Ethical Review: The proposed plan is sent to the SymbolicEvaluator. If the plan violates the AGI's core value_system, it is rejected.
Trust-Based Delegation: For each ActionStep, the ExecutionUnit:
Identifies the required persona (e.g., coder).
Retrieves all available agents with that persona from the DynamicAgentPool.
Selects the agent with the highest trust_score.
Publishes a MessageModel to the MessageBus for the selected agent.
Agent Execution: An Agent receives and executes the task.
Outcome Processing & Feedback: The ExecutionUnit receives the result.
On Success: The agent's trust_score is increased. The AGI's joy emotion is increased.
On Failure: The agent's trust_score is decreased. The AGI's frustration emotion is increased. A self-mutation analysis is triggered.
State Update: The ExecutionUnit updates the workspace, agent state, and SymbolicMemory.
Loop: The cycle continues.
Meta-Cognitive Loop
Running in parallel to the main execution cycle is the MetaCognitionUnit. This unit periodically wakes up and performs self-improvement tasks that are not tied to a specific goal, such as:

Consolidating memories (compress_episodic_memory).
Generating new goals from internal drives (generate_goal_from_drives).
Pruning its own reasoning mutations (prune_mutations).
Documenting its own learned skills (document_undocumented_skills).
This background process ensures the AGI is constantly reflecting and improving, independent of its active tasks.

---
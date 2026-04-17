# Vibeframe Agent Architecture

Vibeframe uses a multi-agent orchestrated pipeline to handle user intake, generate designs, and iteratively refine them. The core orchestration loop for building the UI is built using **LangGraph**. Below is a detailed breakdown of the framework, nodes, prompts, and workflow.

## Frameworks & Models
- **Framework**: [LangGraph](https://python.langchain.com/v0.2/docs/langgraph/) combined with LangChain.
- **LLM Models**: 
  - `llama-3.3-70b-versatile` (via Groq API) - Used as the heavy-duty Designer and Intake agent.
  - `llama-3.1-8b-instant` (via Groq API) - Used for the fast Critic agent.

---

## 1. Pre-Graph Agents (Intake Phase)
Before entering the LangGraph loop, when a user provides an initial brief (especially via voice), two LLM calls are orchestrated sequentially to set up the design context:

### **Intake Agent**
Analyzes the user's brief to detect the website genre and generate clarifying questions.
- **Prompt:**
  ```text
  You are Vibeframe's Intake Agent. Analyze the user's website design brief.
  1. Detect the website genre from: saas, ecommerce, portfolio, agency, hospitality, healthcare, finance, education, gaming, general
  2. Generate exactly 2 concise, specific clarifying questions tailored to this product
  Return strict JSON only...
  ```

### **Palette Agent**
Generates 3 tailored color directions based on the brief and detected genre.
- **Prompt:**
  ```text
  You are Vibeframe's Palette Agent. Create 3 distinct, tailored color palette directions for a website.
  Return strict JSON only...
  ```

---

## 2. Core LangGraph Workflow (Building Phase)

Once the user approves a palette, the system enters the main LangGraph state machine.

### **State Definition (`PipelineState`)**
The graph passes a shared state including:
- `brief`: The user's input + the approved palette colors.
- `current_html`: The current state of the generated DOM.
- `round`: Tracks design iterations (max 3 rounds).
- `critique`: The parsed JSON output from the Critic Agent detailing issues, suggestions, and a score.

### **Nodes**

#### **Node 1: `designer_node`**
The entry point of the graph. It generates the initial HTML website layout based on the brief and the injected palette colors. It uses the heavy-duty Llama 3 70B model.
- **Prompt Template (`DESIGNER_SYSTEM_PROMPT`):**
  ```text
  You are Vibeframe's Elite Senior UI Designer. You create STUNNING, production-quality landing pages rendered as HTML with inline styles on Paper Design canvas.
  
  Required page structure (ALL sections mandatory, in this order):
  1. Navigation bar
  2. Hero section
  3. Social proof / trust bar
  4. Features section
  5. How it works
  6. CTA section
  7. Footer
  
  CRITICAL PAPER MCP RULES:
  1. ALL styles must be inline on every HTML element — no <style> tags, no class-based CSS
  2. The html field must be ONE root <div> containing all sections
  ...
  ```

#### **Node 2: `critic_node`**
Evaluates the designer's HTML output. It checks for visual hierarchy, missing sections, and adherence to the layout rules, outputting a score out of 10.
- **Prompt Template (`CRITIC_SYSTEM_PROMPT`):**
  ```text
  You are Vibeframe Critic Agent.
  Evaluate current landing-page quality and return strict JSON only:
  {
    "score": number,
    "issues": string[],
    "suggestions": string[]
  }
  
  Scoring rubric:
  - 1-3: broken, missing sections, poor hierarchy
  - 4-6: functional but weak aesthetics and layout
  - 7-8: strong baseline with minor polish gaps
  - 9-10: excellent, production-quality visual system and hierarchy
  ```

#### **Node 3: `designer_refine_node`**
If the critic's score is too low, this node takes the current HTML and the critic's `suggestions` to dynamically update and refine the design.

---

## 3. LangGraph Execution Workflow

The iteration logic is mapped via a `StateGraph`:

1. **START** 👉 `designer_node`
2. `designer_node` 👉 `critic_node`
3. **Conditional Edge (`_should_continue`)** at `critic_node`:
   - If `score < target_score` AND `round < max_rounds` 👉 Routes to `designer_refine_node` (**Refine Path**)
   - If `score >= target_score` OR `round >= max_rounds` 👉 Routes to **END** (**Success/Exit Path**)
   - If the JSON evaluation parsing completely failed 👉 Routes to `critic_node` again (**Retry Path**)
4. `designer_refine_node` 👉 Always routes back to `critic_node` to re-evaluate the fixes.

During this loop, the backend utilizes `AgentEventBroker` to **stream real-time SSE events** (e.g., `design_started`, `critic_thinking`, `refine_completed`) down to the frontend UI, powering the agent bubbles you see on the dashboard.
